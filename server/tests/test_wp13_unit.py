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


# ------------------------------------------------------------------
# B: 発話の型と、型ごとの検索の範囲
# ------------------------------------------------------------------

import re

import pytest

from src.config import SERVER_ROOT, DialogueConfig
from src.intents import classify_intent, load_intent_rules, nouns
from src.knowledge import Reference, SearchResult
from src.procedures import load_procedures
from tests.procedure_helpers import seeded_procedures

RULES = load_intent_rules(SERVER_ROOT / "context" / "intents.yaml")
PROCS = load_procedures(SERVER_ROOT / "context")
QRE = re.compile(KnowledgeConfig().intent_question_pattern)


@pytest.mark.parametrize(("text", "intent"), [
    # procedure（別名表・手続きの語）
    ("国保のことで聞きに来ました", "procedure"), ("住民票を取りたいんですが", "procedure"),
    ("児童手当について", "procedure"), ("ごみの出し方を教えて", "procedure"), ("印鑑登録の手続きはどこ", "procedure"),
    # facility
    ("トイレはどこですか", "facility"), ("駐車場ありますか", "facility"), ("何時まで開いてますか", "facility"),
    ("エレベーターは", "facility"), ("授乳室はありますか", "facility"),
    # event
    ("今週末なにかありますか", "event"), ("秋祭りはいつですか", "event"), ("料理教室ってある？", "event"),
    ("イベントの予定は", "event"), ("花火大会はありますか", "event"),
    # tourism
    ("おすすめの観光地は", "tourism"), ("近くに温泉ある？", "tourism"), ("道の駅はどんなところ", "tourism"),
    ("お昼ご飯を食べるところは", "tourism"), ("お土産を買いたい", "tourism"),
    # weather
    ("今日は雨が降りますか", "weather"), ("いい天気ですね", "weather"), ("暑いですね", "weather"),
    ("傘いりますか", "weather"), ("明日の天気は", "weather"),
    # smalltalk（短い返事・報告・相槌）
    ("わかりました", "smalltalk"), ("うん", "smalltalk"), ("うんまだ知ってないです", "smalltalk"),
    ("孫が今度遊びに来るんです", "smalltalk"), ("ありがとう、助かりました", "smalltalk"),
    # other（どれでもない問い・名詞だけの短い発話）
    ("あなたはロボットですか", "other"), ("町からの何か案内ってありますか", "other"), ("印鑑登録", "procedure"),
    ("協調学習ってなに", "other"), ("江田島の人口は", "other"),
])
def test_classify_intent(text: str, intent: str) -> None:
    assert classify_intent(text, RULES, PROCS, None, QRE) == intent


def test_intent_priority() -> None:
    # 手続き ＞ 場所 ＞ 行事 ＞ 観光 ＞ 天気
    assert classify_intent("国保の窓口は何階ですか", RULES, PROCS, None, QRE) == "procedure"
    assert classify_intent("イベントの日の駐車場は", RULES, PROCS, None, QRE) == "facility"
    assert classify_intent("道の駅のイベントは", RULES, PROCS, None, QRE) == "event"
    assert classify_intent("雨の日のおすすめの観光地", RULES, PROCS, None, QRE) == "tourism"


def test_tourism_alias_of_draft_is_not_procedure(tmp_path: Path) -> None:
    # 下書き「観光案内」の別名「観光」「観光地」だけで当たった問いは観光（調べるの提案が先）。ほかの語が当たれば手続き
    from src.procedure_kb import ProcedureBook, ProcedureRecord

    ProcedureBook(tmp_path).save(ProcedureRecord(id="kanko", name="観光案内", aliases=("観光", "観光地"),
                                                 department="商工観光課", status="draft"))
    procs = load_procedures(SERVER_ROOT / "context", tmp_path)
    assert classify_intent("江田島でおすすめの観光地はありますか", RULES, procs, None, QRE) == "tourism"
    assert classify_intent("このあたりの観光の案内はどちらで聞けますか", RULES, procs, None, QRE) == "tourism"
    assert classify_intent("観光の補助金の申請をしたい", RULES, procs, None, QRE) == "procedure"
    assert classify_intent("住民票を取りたいんですが", RULES, procs, None, QRE) == "procedure"


def test_nouns() -> None:
    assert nouns("お祭りはありますか") == ["祭り"]
    assert nouns("町からの何か案内ってありますか") == ["案内"]
    assert nouns("わかりました") == []


class _KB:
    def __init__(self, refs: list[Reference]) -> None:
        self.refs, self.calls = refs, []  # type: ignore[var-annotated]

    def search(self, query: str, uses: Any = None) -> SearchResult:
        self.calls.append(uses)
        return SearchResult(list(self.refs), 5.0)


class _LLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, *a: Any, **k: Any) -> str:
        return ""

    def chat_stream(self, *a: Any, **k: Any) -> Any:
        yield self.reply


TAX = Reference("国民健康保険税のQ&A", "u1", "国保税の額は所得で決まります。", 0.9, 1.0)
EVENT = Reference("秋祭り", "u2", "秋祭りは10月25日です。", 0.9, 1.0)
JOB = Reference("町内求人情報", "u3", "通勤手当あり。", 0.9, 1.0)


async def _say(d: Any, text: str) -> list[str]:
    return [s async for s in d.respond(text)]


async def test_approved_procedure_turn_gets_no_pack(caplog: Any) -> None:
    from src.dialogue import Dialogue

    caplog.set_level("INFO")
    kb = _KB([TAX])
    d = Dialogue(_LLM("保健医療課で手続きできます。"), DialogueConfig(), knowledge=kb, procedures=seeded_procedures())
    await _say(d, "国保のことで聞きに来ました")
    assert d.last_intent == "procedure" and kb.calls == []  # 承認済み（国保の加入）に当たった: パックを引かない
    assert all(r.trust == "procedure" for r in d.last_references)
    assert any("型: procedure / 渡した資料: 承認済み 1・パック 0・お知らせ 0" in r.message for r in caplog.records)


async def test_smalltalk_facility_tourism_get_no_refs() -> None:
    from src.dialogue import Dialogue

    for text in ("わかりました", "トイレはどこですか", "おすすめの観光地は", "いい天気ですね"):
        kb = _KB([TAX, EVENT, JOB])
        d = Dialogue(_LLM("そうですか。"), DialogueConfig(), knowledge=kb)
        await _say(d, text)
        assert kb.calls == [] and d.last_references == [], text


async def test_other_and_event_need_nouns_and_use_dialogue() -> None:
    from src.dialogue import Dialogue

    kb = _KB([TAX, EVENT, JOB])
    d = Dialogue(_LLM("秋祭りは10月25日です。"), DialogueConfig(), knowledge=kb)
    await _say(d, "秋祭りはいつですか")
    assert kb.calls == [frozenset({"dialogue"})] and [r.title for r in d.last_references] == ["秋祭り"]
    kb2 = _KB([TAX, EVENT, JOB])
    d2 = Dialogue(_LLM("ありません。"), DialogueConfig(), knowledge=kb2)
    await _say(d2, "町からの何か案内ってありますか")
    assert d2.last_references == []  # 「案内」は資料の題名・本文の先頭に無い（道の駅のブログを本題にしない）


async def test_event_includes_news_articles() -> None:
    from src.dialogue import Dialogue

    d = Dialogue(_LLM("秋祭りは10月25日です。"), DialogueConfig(), knowledge=_KB([]))
    d.news_refs = lambda q: [Reference("江田島湾海上花火大会", "n1", "花火大会は10月31日です。", 1.0, 0.0, "news")]
    await _say(d, "花火大会はありますか")
    assert [r.title for r in d.last_references] == ["江田島湾海上花火大会"]


async def test_tourism_proposes_lookup(tmp_path: Path) -> None:
    from src.lookup import LookupResult
    from tests.test_actions_unit import StubLLM, StubLookup, _conv

    c = _conv(StubLLM([["温泉はいくつかありますよ。"]]), tmp_path, StubLookup(LookupResult(found=False)))
    out = [s async for s in c.respond("近くに温泉ある？")]
    assert c.dialogue.last_intent == "tourism" and c.turn_info["proposed"] == "lookup"
    assert out[-1] == "よければ、ちょっと調べてみましょうか？"


async def test_lookup_searches_pack_dialogue_and_lookup_first(tmp_path: Path) -> None:
    from src.config import LookupConfig
    from src.lookup import Lookup

    asked: list[str] = []
    onsen = Reference("温井ダムと温泉", "https://cs-akiota.or.jp/x/", "温泉の案内です。", 0.9, 1.0, "semi", "あきおおたから")

    def pack(q: str) -> list[Reference]:
        asked.append(q)
        return [onsen, Reference("協調学習", "u", "学校の取り組み", 0.9, 1.0)]

    async def make_query(t: str) -> str:
        return "温泉"

    lk = Lookup(LookupConfig(sources_dir=tmp_path), "akiota", tmp_path, make_query, pack=pack)
    result = await lk.run("近くに温泉ある？")
    assert result.found and [r.title for r in result.references] == ["温井ダムと温泉"] and asked == ["温泉"]


async def test_approved_procedure_reply_gets_department_when_missing() -> None:
    # 承認済みの手続きで答えたのに課の名前を言わなかった: 規則で決まった担当の課を一文足す（WP13 C の 30 問）
    from src.dialogue import Dialogue
    from src.site_context import load_facility

    fac = load_facility("etajima", SERVER_ROOT / "context")
    d = Dialogue(_LLM("まずは一時停止の手続きをしてください。"), DialogueConfig(), knowledge=_KB([]),
                 procedures=seeded_procedures(), facility=fac)
    said = await _say(d, "マイナンバーカードをなくしたんですが")
    assert said[-1] == "担当は市民課（1階）です。"
    d = Dialogue(_LLM("市民課（1階）で再交付の手続きができます。"), DialogueConfig(), knowledge=_KB([]),
                 procedures=seeded_procedures(), facility=fac)
    said = await _say(d, "マイナンバーカードをなくしたんですが")
    assert not any(s.startswith("担当は") for s in said)  # 課の名前を言っていれば足さない


async def test_procedure_without_approved_counts_pages_not_chunks() -> None:
    # 承認済みが無い手続きの上位 2 件は、ページで 2 件（同じページの 2 チャンクで埋めない。手続き 100 問の 96 番）
    from src.dialogue import Dialogue

    a1 = Reference("イノシシに注意", "u/a", "イノシシの被害が多発しています。", 0.9, 1.0)
    a2 = Reference("イノシシに注意", "u/a", "市では捕獲活動や補助金を実施しています。", 0.9, 1.0)
    b = Reference("イノシシ110番のご案内", "u/b", "農林水産課内にイノシシ110番があります。", 0.88, 1.0)
    d = Dialogue(_LLM("そうですか。"), DialogueConfig(), knowledge=_KB([a1, a2, b]))
    await _say(d, "イノシシの補助金の申請をしたい")
    assert d.last_intent == "procedure"
    assert [r.source_url for r in d.last_references] == ["u/a", "u/b"]
