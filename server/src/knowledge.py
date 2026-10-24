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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import KnowledgeConfig

logger = logging.getLogger(__name__)

_LANCE_TABLE = "chunks"
JST = timezone(timedelta(hours=9))
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
    # 出どころ（official / semi / news）と、応答で出典を言うときの言い方（「観光協会のホームページ」など）
    trust: str = ""
    label: str = ""
    # 知識パックのチャンクの ID（文字記録に出典として残す。調べるで取ったページは空）
    chunk_id: str = ""


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

    def __init__(self, pack_dir: Path, cfg: KnowledgeConfig, model: Any = None) -> None:
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
        # 用途（WP13）。format 1（WP12 まで）のパックには use が無いので全部 dialogue とみなす
        for r in rows:
            r["use"] = r.get("use") or "dialogue"
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
        self.model_name = str(emb["model"])
        if model is not None and getattr(model, "idobata_name", None) == self.model_name:
            self._model = model  # 差し替えのときは埋め込みモデルを使い回す
        else:
            torch.set_num_threads(cfg.embed_threads)
            self._model = SentenceTransformer(self.model_name, device=cfg.embed_device)
            self._model.idobata_name = self.model_name
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

    def use_counts(self) -> dict[str, int]:
        """用途ごとのチャンク数。"""
        out: dict[str, int] = defaultdict(int)
        for r in self._rows:
            out[str(r["use"])] += 1
        return dict(out)

    def search(self, query: str, uses: frozenset[str] | None = None) -> SearchResult:
        """来訪者の発話で参考資料を探す（同期、CPU）。uses を渡すと、その用途のチャンクだけ（WP13）。"""
        t0 = time.perf_counter()
        cfg = self._cfg
        vec = self._model.encode([self._query_prefix + query], normalize_embeddings=True)[0]
        # 用途で絞るときは、候補を多めに取ってから落とす（同じ順位の付け方のまま）
        widen = 1 if uses is None else cfg.use_filter_widen
        hits = self._vector_query(vec).limit(cfg.vector_candidates * widen).to_list()
        if uses is not None:
            hits = [h for h in hits if self._rows[self._index_of[h["id"]]]["use"] in uses][: cfg.vector_candidates]
        vec_rank = {self._index_of[h["id"]]: (r, 1.0 - float(h["_distance"]))
                    for r, h in enumerate(hits)}
        bm25 = self._bm25.search(query, cfg.bm25_candidates * widen)
        if uses is not None:
            bm25 = [(d, sc) for d, sc in bm25 if self._rows[d]["use"] in uses][: cfg.bm25_candidates]
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
                                  bm_rank[doc][1] if doc in bm_rank else 0.0,
                                  str(row.get("trust") or ""), str(row.get("label") or ""), str(row["id"])))
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


class KnowledgeStore:
    """今の知識パックを持ち、manifest.json が変わったら読み直して差し替える（無停止）。

    ``$DATA/kb/<site>`` はシンボリックリンクで、kb/build.py が新しい版を作ってから付け替える。
    リンク先の実体のパスと manifest.json の更新時刻を見て、変わっていれば新しい版を読む。
    読み込みは呼び出し側が別スレッドで行う（``reload_if_changed``）。検索中の古い版は
    参照が残る限りそのまま使える（ディレクトリは build.py が直前の版まで残す）。
    """

    def __init__(self, data_dir: Path, site: str, cfg: KnowledgeConfig) -> None:
        self._manifest = data_dir / "kb" / site / "manifest.json"
        self._cfg = cfg
        self._site = site
        self.current: KnowledgeBase | None = None
        self._loaded: tuple[str, float] | None = None
        self._warned = False

    def _key(self) -> tuple[str, float] | None:
        try:
            real = self._manifest.resolve(strict=True)
            return str(real), real.stat().st_mtime
        except OSError:
            return None

    def reload_if_changed(self) -> bool:
        """新しい版があれば読み込んで差し替える（数秒かかるので別スレッドで呼ぶ）。"""
        if not self._cfg.enabled:
            return False
        key = self._key()
        if key is None:
            if not self._warned:
                logger.warning("知識パックがありません: %s（uv run --project kb python kb/build.py "
                               "--site %s）", self._manifest.parent, self._site)
                self._warned = True
            return False
        if key == self._loaded:
            return False
        try:
            kb = KnowledgeBase(Path(key[0]).parent, self._cfg,
                               model=self.current._model if self.current else None)
        except Exception:
            logger.exception("新しい知識パックを読み込めません（今の版を使い続けます）")
            return False
        replaced = self.current is not None
        self.current, self._loaded = kb, key
        if replaced:
            logger.info("知識パックを差し替えました: %s", Path(key[0]).parent.name)
        return True

    def search(self, query: str, uses: frozenset[str] | None = None) -> SearchResult:
        kb = self.current  # 差し替えの途中でも、取った版で最後まで検索する
        if kb is None:
            return SearchResult([], 0.0)
        return kb.search(query, uses)


class Weather:
    """weather.json（朝のバッチで差し替わる）を、更新されたら読み直して文にする。

    知識パック本体が無くても天気だけは使えるよう、KnowledgeBase とは別にする。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime = 0.0
        self._data: dict[str, Any] = {}
        self._warned = False

    def lines(self, now: datetime) -> list[str]:
        """今の気温（アメダス）と、今日・明日の予報の文。古いものは出さない。"""
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
        return format_weather(self._data, now)

    def today(self, now: datetime, max_obs_age_sec: float = 3 * 3600) -> tuple[str | None, float | None]:
        """今日の予報の天気（気象庁の文言のまま）と、今の気温（古ければ None）。切り出しの一言に使う（WP02 C）。"""
        self.lines(now)  # 更新されていれば読み直す
        weather = next((str(f.get("weather") or "") for f in self._data.get("forecasts", [])
                        if f.get("date") == now.date().isoformat()), None)
        temp = None
        obs = self._data.get("observation") or {}
        try:
            age = (now.astimezone(JST) - datetime.fromisoformat(obs["observed_at"]).astimezone(JST)).total_seconds()
            if 0 <= age <= max_obs_age_sec:
                temp = float(obs["temp"])
        except (KeyError, ValueError, TypeError):
            pass
        return weather or None, temp

    @classmethod
    def for_site(cls, data_dir: Path, site: str) -> Weather:
        return cls(data_dir / "kb" / site / "weather.json")


def format_weather(
    weather: dict[str, Any], now: datetime, max_obs_age_sec: float = 3 * 3600
) -> list[str]:
    """weather.json を、今の時刻に合わせた文にする。

    - 気温: アメダス最寄り観測点の実況（max_obs_age_sec より古ければ言わない）。
      観測点が拠点の中なら「安芸太田は今…」、外なら「江田島の近くの呉では今…」
    - 予報: 一次細分区域の天気（気象庁の文言のまま）と、これからの降水確率。
      「今日・明日」は発表日ではなく今日の日付で付け直し、古い予報は出さない
    """
    today = now.date()
    labels = {today: "今日", today + timedelta(days=1): "明日"}
    lines: list[str] = []
    obs = weather.get("observation")
    if obs:
        try:
            observed = datetime.fromisoformat(obs["observed_at"]).astimezone(JST)
            age = (now.astimezone(JST) - observed).total_seconds()
        except (KeyError, ValueError, TypeError):
            age = float("inf")
        if 0 <= age <= max_obs_age_sec:
            temp = round(float(obs["temp"]))
            when = f"{observed.hour}時{observed.minute:02d}分の観測"
            if obs.get("in_site"):
                lines.append(f"{obs['place']}は今{temp}度くらいです（{when}）。")
            else:
                lines.append(f"{obs['place']}の近くの{obs['station']}では今{temp}度くらいです（{when}）。")
    for f in weather.get("forecasts", []):
        try:
            d = date.fromisoformat(f["date"])
        except (KeyError, ValueError):
            continue
        if d not in labels:
            continue
        text = f"{labels[d]}（{d.month}月{d.day}日）の{f['region']}は「{f['weather']}」の予報です。"
        pops = [p for p in f.get("pops") or [] if d > today or p["to"] > now.hour]
        if pops and d == today:
            text += "降水確率は、" + "、".join(
                f"{p['from']}時から{p['to']}時が{p['pop']}パーセント" for p in pops) + "です。"
        elif pops:
            text += f"降水確率は最大{max(p['pop'] for p in pops)}パーセントです。"
        lines.append(text)
    if lines and weather.get("source"):
        lines.append(f"（{weather['source']}の発表・観測）")
    return lines
