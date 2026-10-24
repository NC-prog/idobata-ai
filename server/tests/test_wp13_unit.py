"""WP13: 会話で使う知識の絞り込み（パックの用途 use・古い形式の互換・用途で絞る検索）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import KnowledgeConfig
from src.knowledge import KnowledgeBase

_MODEL = "fake-model"


class _FakeModel:
    """語の有無で決まる 4 次元のベクトル（住民票・ごみ・観光・その他）。"""

    idobata_name = _MODEL

    def encode(self, texts: list[str], **kw: Any) -> np.ndarray:
        out = []
        for t in texts:
            v = np.array([("住民票" in t) * 1.0, ("ごみ" in t) * 1.0, ("観光" in t or "道の駅" in t) * 1.0, 0.1])
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def _make_pack(path: Path, rows: list[tuple[str, str, str | None]], fmt: int) -> None:
    """kb/build.py と同じ形のパック。fmt=1 は use の列が無い（WP12 まで）。"""
    import lancedb
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path.mkdir(parents=True)
    ids = [f"c{i}" for i in range(len(rows))]
    vecs = _FakeModel().encode([f"{t}\n{x}" for t, x, _ in rows])
    cols: dict[str, Any] = {"id": ids, "source_url": [f"https://example.jp/{i}" for i in ids],
                            "title": [t for t, _, _ in rows], "fetched_at": ["2026-10-25"] * len(rows),
                            "text": [x for _, x, _ in rows]}
    lance: dict[str, Any] = {"id": ids}
    if fmt >= 2:
        cols["use"] = lance["use"] = [u for _, _, u in rows]
    pq.write_table(pa.table(cols), path / "chunks.parquet")
    lance["vector"] = pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), 4))
    lancedb.connect(str(path / "index.lancedb")).create_table("chunks", data=pa.table(lance))
    manifest: dict[str, Any] = {"embedding": {"model": _MODEL, "query_prefix": ""}}
    if fmt >= 2:
        manifest["format"] = 2
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


ROWS = [("住民票の写し", "住民票の写しは市民課で交付します。", "dialogue"),
        ("ごみの出し方", "燃やせるごみは週2回です。", "dialogue"),
        ("道の駅の観光", "道の駅で観光の案内をしています。住民票は扱いません。", "lookup")]
CFG = KnowledgeConfig(min_vector_score=0.0, min_vector_score_with_keyword=0.0)


def test_old_format_pack_reads_as_dialogue(tmp_path: Path) -> None:
    _make_pack(tmp_path / "p", ROWS, fmt=1)
    kb = KnowledgeBase(tmp_path / "p", CFG, model=_FakeModel())
    assert kb.use_counts() == {"dialogue": 3}
    titles = [r.title for r in kb.search("住民票", frozenset({"dialogue"})).references]
    assert "道の駅の観光" in titles  # 古い形式は全部 dialogue（用途で絞っても落ちない）


def test_search_filters_by_use(tmp_path: Path) -> None:
    _make_pack(tmp_path / "p", ROWS, fmt=2)
    kb = KnowledgeBase(tmp_path / "p", CFG, model=_FakeModel())
    assert kb.use_counts() == {"dialogue": 2, "lookup": 1}
    every = [r.title for r in kb.search("道の駅の観光と住民票").references]
    assert "道の駅の観光" in every  # 絞らなければ全部
    talk = [r.title for r in kb.search("道の駅の観光と住民票", frozenset({"dialogue"})).references]
    assert "道の駅の観光" not in talk and "住民票の写し" in talk
    look = [r.title for r in kb.search("道の駅の観光", frozenset({"lookup"})).references]
    assert look == ["道の駅の観光"]
