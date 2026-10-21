"""WP03: 拠点の固定事実・システムプロンプト・知識パック検索のテスト。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from src.config import SERVER_ROOT, DialogueConfig, KnowledgeConfig
from src.knowledge import _BM25, JST, KnowledgeBase, Reference, bigrams, format_weather
from src.persona import (
    build_system_prompt,
    build_turn_message,
    format_department,
    organization_word,
)
from src.site_context import Department, Facility, load_facility, load_site_words


class TestFacility:
    def test_etajima_profile(self) -> None:
        fac = load_facility("etajima", SERVER_ROOT / "context")
        assert fac is not None
        assert fac.name == "江田島市役所"
        depts = {d.name: d for d in fac.departments}
        assert depts["市民課"].floor == "1階"
        assert "住民票" in depts["市民課"].handles
        assert "国民健康保険" in depts["保健医療課"].handles
        assert depts["子育て支援課"].floor is None and depts["子育て支援課"].place
        assert fac.emergency_lines[0].number == "0120-95-0178"
        assert "LINE" in fac.allow_ascii
        assert "宮島" in fac.outside_places

    def test_akiota_profile_has_no_floors(self) -> None:
        fac = load_facility("akiota", SERVER_ROOT / "context")
        assert fac is not None
        assert fac.name == "安芸太田町役場"
        assert all(d.floor is None for d in fac.departments)  # 階は公式サイトで確認できなかった
        assert any("戸河内" in n for n in fac.notes)

    def test_facility_is_not_vocabulary_but_department_names_are(self) -> None:
        words = {w.surface for w in load_site_words("etajima", SERVER_ROOT / "context")}
        assert "市民課" in words and "保健医療課" in words
        assert "江田島市役所の1階ロビー" not in words
        assert "LINE" not in words
        assert "宮島" not in words
        assert "市民生活課" not in words  # 旧・仮の課名は外した

    def test_missing_facility(self, tmp_path: Path) -> None:
        (tmp_path / "x.yaml").write_text("site_name: テスト市\n", encoding="utf-8")
        assert load_facility("x", tmp_path) is None


class TestPersona:
    def test_department_formats(self) -> None:
        assert format_department(Department("市民課", "1階", None, ("住民票", "戸籍"))) == (
            "市民課（1階）: 住民票、戸籍"
        )
        assert format_department(Department("建設課", "4階")) == "建設課（4階）"
        assert "この建物ではなく 別館" in format_department(Department("子育て支援課", place="別館"))

    def test_prompt_contains_facts_references_and_today(self) -> None:
        fac = load_facility("etajima", SERVER_ROOT / "context")
        refs = [Reference("秋祭りのお知らせ", "https://example.jp/1", "10月25日に開催", 0.9, 3.0)]
        prompt = build_system_prompt(DialogueConfig(), fac, date(2026, 10, 21), ["今日は「晴れ」の予報です。"])
        turn = build_turn_message(DialogueConfig(), fac, refs, "お祭りは？")
        assert prompt.startswith("あなたは江田島市役所の1階ロビーにいる案内役です。")
        assert "市民課（1階）: 戸籍" in prompt
        assert "0120-95-0178" in prompt
        assert "[1] (出典: 秋祭りのお知らせ)\n10月25日に開催" in turn  # 出どころの無い古いパック
        assert turn.endswith("【来訪者の発話】\nお祭りは？")
        assert "2026年10月21日（水曜日）。今週末は10月24日（土曜日）と10月25日（日曜日）" in prompt
        assert "今日は「晴れ」の予報です。" in prompt
        assert "お住まいの市区町村" in prompt
        assert "[出典:1,3]" in turn

    def test_source_labels(self) -> None:
        from src.persona import source_labels

        refs = [Reference("a", "u", "t", 0.9, 1.0, "semi", "観光協会のホームページ"),
                Reference("b", "u", "t", 0.9, 1.0),
                Reference("c", "u", "t", 0.9, 1.0, "semi", "観光協会のホームページ")]
        fac = Facility("安芸太田町役場", "ロビー", address_city="安芸太田町")
        assert source_labels(DialogueConfig(), fac, refs) == "観光協会のホームページや町のホームページ"

    def test_org_word(self) -> None:
        assert organization_word(Facility("安芸太田町役場", "ロビー", address_city="安芸太田町")) == "町"
        assert organization_word(None) == "市"


class TestWeather:
    _W: ClassVar[dict[str, Any]] = {
        "source": "気象庁（広島地方気象台）",
        "forecasts": [
            {"date": "2026-10-21", "region": "広島県南部", "weather": "雨　夜　くもり",
             "pops": [{"from": 12, "to": 18, "pop": 70}, {"from": 18, "to": 24, "pop": 20}]},
            {"date": "2026-10-22", "region": "広島県南部", "weather": "晴れ",
             "pops": [{"from": 0, "to": 6, "pop": 0}, {"from": 6, "to": 12, "pop": 10}]},
        ],
        "observation": {"station_id": "67511", "station": "呉", "place": "江田島",
                        "in_site": False, "temp": 23.0, "observed_at": "2026-10-21T15:40+09:00"},
    }

    def test_now_temperature_and_forecast(self) -> None:
        lines = format_weather(self._W, datetime(2026, 10, 21, 16, 0, tzinfo=JST))
        assert lines[0] == "江田島の近くの呉では今23度くらいです（15時40分の観測）。"
        assert lines[1] == ("今日（10月21日）の広島県南部は「雨　夜　くもり」の予報です。"
                            "降水確率は、12時から18時が70パーセント、18時から24時が20パーセントです。")
        later = format_weather(self._W, datetime(2026, 10, 21, 18, 30, tzinfo=JST))
        assert "12時から18時" not in later[1]  # 過ぎた時間帯は言わない
        assert lines[2] == "明日（10月22日）の広島県南部は「晴れ」の予報です。降水確率は最大10パーセントです。"
        assert lines[3] == "（気象庁（広島地方気象台）の発表・観測）"
        assert not any("広島の" in x or "庄原" in x for x in lines)  # 代表点の地名は言わない

    def test_in_site_station_and_stale_observation(self) -> None:
        w = {**self._W, "observation": {**self._W["observation"], "station": "加計",
                                         "place": "安芸太田", "in_site": True}}
        assert format_weather(w, datetime(2026, 10, 21, 16, 0, tzinfo=JST))[0] == "安芸太田は今23度くらいです（15時40分の観測）。"
        stale = format_weather(w, datetime(2026, 10, 21, 20, 0, tzinfo=JST))  # 4 時間前の観測は言わない
        assert not stale[0].startswith("安芸太田は今")

    def test_relabels_by_current_date_and_drops_old(self) -> None:
        lines = format_weather(self._W, datetime(2026, 10, 22, 8, 0, tzinfo=JST))
        assert lines[0].startswith("今日（10月22日）")
        assert format_weather(self._W, datetime(2026, 11, 1, 8, 0, tzinfo=JST)) == []


def test_bm25_prefers_documents_sharing_bigrams() -> None:
    bm = _BM25(["住民票の写しは市民課で交付", "ごみの出し方", "国民健康保険の加入"], 1.5, 0.75)
    hits = bm.search("住民票はどこ", 3)
    assert hits[0][0] == 0
    assert bigrams("住民 票。") == ["住民", "民票"]


_MODEL = "cl-nagoya/ruri-v3-30m"


def _make_pack(path: Path, texts: list[tuple[str, str]]) -> None:
    """小さな知識パックを作る（kb/build.py と同じ形式）。"""
    import json

    import lancedb
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer

    path.mkdir(parents=True)
    model = SentenceTransformer(_MODEL, device="cpu")
    vecs = model.encode([f"検索文書: {t}\n{x}" for t, x in texts], normalize_embeddings=True)
    ids = [f"{path.name}-{i}" for i in range(len(texts))]
    pq.write_table(pa.table({
        "id": ids, "source_url": [f"https://example.jp/{i}" for i in ids],
        "title": [t for t, _ in texts], "fetched_at": ["2026-10-21"] * len(texts),
        "text": [x for _, x in texts],
    }), path / "chunks.parquet")
    lancedb.connect(str(path / "index.lancedb")).create_table("chunks", data=pa.table({
        "id": ids, "vector": pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), 256)),
    }))
    (path / "manifest.json").write_text(json.dumps({"embedding": {
        "model": _MODEL, "query_prefix": "検索クエリ: "}}), encoding="utf-8")


@pytest.mark.integration
def test_knowledge_base_hybrid_search(tmp_path: Path) -> None:
    """小さな知識パックを実際に作って検索する（Ruri v3 の重みが必要なので integration）。"""
    _make_pack(tmp_path / "p", [
        ("住民票の写し", "住民票の写しは市民課の窓口で交付します。本人確認書類をお持ちください。"),
        ("ごみの出し方", "燃やせるごみは週2回、指定袋に入れて朝8時までに出してください。"),
        ("秋祭り", "10月25日の土曜日に能美の海辺で秋祭りを開催します。"),
    ])
    tmp_path = tmp_path / "p"
    kb = KnowledgeBase(tmp_path, KnowledgeConfig(min_vector_score=0.0))
    result = kb.search("住民票はどこで取れますか")
    assert result.references[0].title == "住民票の写し"
    assert result.elapsed_ms < 1000
    strict = KnowledgeBase(tmp_path, KnowledgeConfig(min_vector_score=0.99))
    assert strict.search("今日は暖かいですね").references == []


@pytest.mark.integration
def test_knowledge_store_swaps_pack_without_restart(tmp_path: Path) -> None:
    """シンボリックリンクを付け替えると、読み直して新しい版で検索する（モデルは使い回す）。"""
    import os

    from src.knowledge import KnowledgeStore

    versions = tmp_path / "kb" / "etajima.versions"
    _make_pack(versions / "v1", [("ごみの出し方", "燃やせるごみは週2回、指定袋に入れて出してください。")])
    _make_pack(versions / "v2", [("秋祭り", "10月31日に江田島湾で花火大会を開催します。")])
    link = tmp_path / "kb" / "etajima"
    link.symlink_to(versions / "v1", target_is_directory=True)
    store = KnowledgeStore(tmp_path, "etajima", KnowledgeConfig(min_vector_score=0.0))
    assert store.reload_if_changed() is True
    assert store.reload_if_changed() is False  # 変わっていなければ読み直さない
    assert store.search("花火はいつ").references[0].title == "ごみの出し方"
    model = store.current._model if store.current else None
    tmp = tmp_path / "kb" / ".swap"
    tmp.symlink_to(versions / "v2", target_is_directory=True)
    os.replace(tmp, link)
    assert store.reload_if_changed() is True
    assert store.search("花火はいつ").references[0].title == "秋祭り"
    assert store.current is not None and store.current._model is model
