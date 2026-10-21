"""知識パックの検索（WP03）。

``kb/build.py`` が作ったディレクトリ（``$IDOBATA_DATA_DIR/kb/<site>/``）を読む:

- ``chunks.parquet``（本文・出典）と ``index.lancedb``（埋め込みベクトル）
- ``manifest.json``（埋め込みモデル名・プレフィックス）
- ``weather.json``（気象庁の予報。朝のバッチで差し替わるので、更新されたら読み直す。``Weather``）

検索はベクトル検索（Ruri v3、CPU）と BM25（文字 2-gram）のハイブリッド。両方の順位を
RRF で融合し、ベクトルの類似度がしきい値未満の資料は落とす（関係ない資料を渡さない）。
サーバプロセスは立てない（LanceDB は埋め込み型）。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .config import KnowledgeConfig

logger = logging.getLogger(__name__)

_LANCE_TABLE = "chunks"
# BM25 の文字 2-gram を作る前に落とす文字（空白・記号）
_NON_TEXT_RE = re.compile(r"[\s#|*、。，．,.・「」『』（）()【】［］\[\]！？!?：:；;／/〜~ー－\-]+")


@dataclass(frozen=True)
class Reference:
    """参考資料 1 件。"""

    title: str
    source_url: str
    text: str
    vector_score: float  # cos 類似度
    bm25_score: float


@dataclass(frozen=True)
class SearchResult:
    references: list[Reference]
    elapsed_ms: float


def bigrams(text: str) -> list[str]:
    """記号・空白を除いた文字 2-gram。"""
    t = _NON_TEXT_RE.sub("", text)
    return [t[i : i + 2] for i in range(len(t) - 1)]


class _BM25:
    """文字 2-gram の BM25（日本語の分かち書きをせずに語の一致を拾う）。"""

    def __init__(self, docs: list[str], k1: float, b: float) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, doc in enumerate(docs):
            grams = bigrams(doc)
            self.lengths.append(len(grams))
            for g, tf in Counter(grams).items():
                self.postings[g].append((i, tf))
        self.n = len(docs)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for g in set(bigrams(query)):
            posting = self.postings.get(g)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc, tf in posting:
                norm = 1 - self.b + self.b * self.lengths[doc] / (self.avg_len or 1)
                scores[doc] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda x: -x[1])[:limit]


class KnowledgeBase:
    """知識パック 1 拠点ぶん。"""

    def __init__(self, pack_dir: Path, cfg: KnowledgeConfig) -> None:
        import lancedb
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
        import torch
        from sentence_transformers import SentenceTransformer

        self._cfg = cfg
        self.pack_dir = pack_dir
        self.manifest: dict[str, Any] = json.loads(
            (pack_dir / "manifest.json").read_text(encoding="utf-8")
        )
        rows = pq.read_table(pack_dir / "chunks.parquet").to_pylist()
        self._rows: list[dict[str, Any]] = rows
        self._index_of = {r["id"]: i for i, r in enumerate(rows)}
        self._bm25 = _BM25([f"{r['title']}\n{r['text']}" for r in rows], cfg.bm25_k1, cfg.bm25_b)
        self._table = lancedb.connect(str(pack_dir / "index.lancedb")).open_table(_LANCE_TABLE)
        # BM25 だけで拾った資料の類似度を計算するために、ベクトルを行の順に持っておく（数千件なら数 MB）
        import numpy as np

        vec_rows = self._table.to_arrow().to_pylist()
        order = {r["id"]: r["vector"] for r in vec_rows}
        self._vectors = np.array([order[r["id"]] for r in rows], dtype=np.float32)
        emb = self.manifest["embedding"]
        self._query_prefix = str(emb.get("query_prefix", ""))
        torch.set_num_threads(cfg.embed_threads)
        self._model = SentenceTransformer(str(emb["model"]), device=cfg.embed_device)
        self._model.encode([self._query_prefix + "ウォームアップ"])
        logger.info(
            "知識パック: %s（%d チャンク、%s 生成、埋め込み %s）",
            pack_dir, len(rows), self.manifest.get("generated_at"), emb["model"],
        )

    @classmethod
    def load(cls, data_dir: Path, site: str, cfg: KnowledgeConfig) -> KnowledgeBase | None:
        """知識パックがあれば読む。無ければ警告して None（参考資料なしで動く）。"""
        pack = data_dir / "kb" / site
        if not cfg.enabled:
            return None
        if not (pack / "manifest.json").is_file():
            logger.warning(
                "知識パックがありません: %s（uv run --project kb python kb/build.py --site %s）",
                pack, site,
            )
            return None
        return cls(pack, cfg)

    def search(self, query: str) -> SearchResult:
        """来訪者の発話で参考資料を探す（同期、CPU）。"""
        t0 = time.perf_counter()
        cfg = self._cfg
        vec = self._model.encode([self._query_prefix + query], normalize_embeddings=True)[0]
        hits = self._vector_query(vec).limit(cfg.vector_candidates).to_list()
        vec_rank = {self._index_of[h["id"]]: (r, 1.0 - float(h["_distance"]))
                    for r, h in enumerate(hits)}
        bm25 = self._bm25.search(query, cfg.bm25_candidates)
        bm_rank = {doc: (r, s) for r, (doc, s) in enumerate(bm25)}

        fused: dict[int, float] = defaultdict(float)
        for doc, (r, _) in vec_rank.items():
            fused[doc] += 1 / (cfg.rrf_k + r + 1)
        for doc, (r, _) in bm_rank.items():
            fused[doc] += 1 / (cfg.rrf_k + r + 1)

        refs: list[Reference] = []
        total = 0
        for doc, _ in sorted(fused.items(), key=lambda x: -x[1]):
            vscore = vec_rank[doc][1] if doc in vec_rank else self._cosine(vec, doc)
            keyword_hit = doc in bm_rank and bm_rank[doc][0] < cfg.keyword_rank
            if not (vscore >= cfg.min_vector_score
                    or (keyword_hit and vscore >= cfg.min_vector_score_with_keyword)):
                continue
            row = self._rows[doc]
            text = row["text"]
            if total + len(text) > cfg.max_chars:
                text = text[: max(0, cfg.max_chars - total)]
            if not text:
                break
            refs.append(Reference(row["title"], row["source_url"], text, vscore,
                                  bm_rank[doc][1] if doc in bm_rank else 0.0))
            total += len(text)
            if len(refs) >= cfg.top_k or total >= cfg.max_chars:
                break
        return SearchResult(refs, (time.perf_counter() - t0) * 1000)

    def _vector_query(self, vec: Any) -> Any:
        """cos 距離のベクトル検索（LanceVectorQueryBuilder）。"""
        from lancedb.query import LanceVectorQueryBuilder

        q = self._table.search(vec.tolist(), query_type="vector")
        assert isinstance(q, LanceVectorQueryBuilder)
        return q.distance_type("cosine")

    def _cosine(self, vec: Any, doc: int) -> float:
        """BM25 だけで拾った資料のベクトル類似度（しきい値の判定に使う。ベクトルは正規化済み）。"""
        return float(self._vectors[doc] @ vec)


class Weather:
    """weather.json（朝のバッチで差し替わる）を、更新されたら読み直して文にする。

    知識パック本体が無くても天気だけは使えるよう、KnowledgeBase とは別にする。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime = 0.0
        self._data: dict[str, Any] = {}
        self._warned = False

    def lines(self, today: date) -> list[str]:
        """今日・明日の予報の文（気象庁の文言のまま）。古い予報は出さない。"""
        try:
            mtime = self.path.stat().st_mtime
            if mtime != self._mtime:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime = mtime
        except (OSError, json.JSONDecodeError):
            if not self._warned:
                logger.warning("weather.json を読めません: %s", self.path)
                self._warned = True
            return []
        return format_weather(self._data, today)

    @classmethod
    def for_site(cls, data_dir: Path, site: str) -> Weather:
        return cls(data_dir / "kb" / site / "weather.json")


def format_weather(weather: dict[str, Any], today: date) -> list[str]:
    """weather.json の予報を、今日・明日の文にする（発表日ではなく今日の日付で「今日」を決める）。"""
    labels = {today: "今日", today + timedelta(days=1): "明日"}
    lines: list[str] = []
    for f in weather.get("forecasts", []):
        try:
            d = date.fromisoformat(f["date"])
        except (KeyError, ValueError):
            continue
        if d not in labels:
            continue
        text = f"{labels[d]}（{d.month}月{d.day}日）の{f['region']}は「{f['weather']}」"
        temps = []
        if f.get("max"):
            temps.append(f"最高気温は{f['max']}度")
        if f.get("min"):
            temps.append(f"最低気温は{f['min']}度")
        if temps:
            text += f"、{f['point']}の" + "、".join(temps)
        lines.append(text + "の予報です。")
    if lines and weather.get("source"):
        lines.append(f"（{weather['source']}の発表）")
    return lines
