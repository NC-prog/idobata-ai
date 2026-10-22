"""WP06: 会話の文字記録・管理者ページ・誤り探しの素材・呼び名の語彙・要約の窓・設備の作り話・雑談の資料。"""

from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from src.config import SERVER_ROOT, AdminConfig, DialogueConfig, StaffConfig, build_config
from src.dialogue import Dialogue
from src.edge_server import AdminPanel, StaffPanel, create_app
from src.knowledge import JST, Reference, SearchResult
from src.procedures import AsrFixes
from src.referrals import ReferralStore, VisitLog
from src.site_context import BuildingFacility, Department, Facility, avatar_word, load_site_words
from src.transcripts import TranscriptStore
from tests.test_actions_unit import PHRASED, R, SpyLLM, _conv, _refer_json, _saved, _say
from tests.test_dialogue_unit import StubLLM, _collect

REPO_ROOT = SERVER_ROOT.parent


# ------------------------------------------------------------------
# 1) 会話の文字記録と管理者ページ
# ------------------------------------------------------------------


def _write_day(store: TranscriptStore, d: date, sid: str, n: int = 2) -> None:
    store.root.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": f"{d.isoformat()}T10:00:0{i}+09:00", "session_id": sid, "edge_id": "", "speaker": "user",
             "text": f"発話{i}"} for i in range(n)]
    with (store.root / f"{d.isoformat()}.jsonl").open("a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def test_transcript_append_sessions_and_retention_with_hold(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path, "etajima", 7)
    store.append("user", "s1", "edge-1", asr_raw="国宝の", text="国保の")
    store.append("assistant", "s1", "edge-1", text="保健医療課です。", dropped=[])
    today = datetime.now(JST).date()
    (s1,) = store.sessions(today)
    assert s1["session_id"] == "s1" and s1["lines"] == 2 and s1["user_lines"] == 1 and not s1["held"]
    assert [r["speaker"] for r in store.session(today, "s1")] == ["user", "assistant"]
    assert (tmp_path / "transcripts" / "etajima" / f"{today.isoformat()}.jsonl").is_file()

    old, edge = today - timedelta(days=8), today - timedelta(days=7)
    _write_day(store, old, "old-a")
    _write_day(store, old, "old-held")
    _write_day(store, edge, "edge-day")
    assert store.set_hold("old-held", True, "不具合の調べ")
    assert not store.set_hold("no-such", True)
    removed = store.cleanup(today)
    assert removed == 2  # 8 日前の保全していない行だけ消える（7 日前は残る）
    assert [s["session_id"] for s in store.sessions(old)] == ["old-held"]
    assert store.sessions(old)[0]["held"]
    assert store.sessions(edge)
    store.set_hold("old-held", False)
    store.cleanup(today)
    assert old.isoformat() not in store.days()


def test_transcript_retention_from_env() -> None:
    cfg = build_config({"IDOBATA_TRANSCRIPT_RETENTION_DAYS": "3", "IDOBATA_ADMIN_PASSWORD": " pw "})
    assert cfg.transcript.retention_days == 3 and cfg.admin.password == "pw" and cfg.admin.user == "admin"
    assert build_config({}).transcript.retention_days == 7


async def test_admin_transcripts_is_separate_role(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path, "etajima", 7)
    store.append("user", "s1", "", asr_raw="こんにちは", text="こんにちは")
    staff = StaffPanel(ReferralStore(tmp_path, "etajima", 30, csv_export=False), VisitLog(tmp_path, "etajima"),
                       StaffConfig(password="staffpw"))
    admin = AdminPanel(store, AdminConfig(password="adminpw"))
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none", staff, admin), client=("192.168.1.20", 5000))
    day = datetime.now(JST).date().isoformat()
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/admin/transcripts/")).status_code == 401
        assert (await c.get("/admin/transcripts/", auth=("staff", "staffpw"))).status_code == 401  # 職員では読めない
        assert (await c.get("/staff/", auth=("admin", "adminpw"))).status_code == 401
        page = await c.get("/admin/transcripts/", auth=("admin", "adminpw"))
        assert page.status_code == 200 and "会話の文字記録" in page.text and "7 日" in page.text
        rows = (await c.get(f"/admin/api/transcripts?day={day}", auth=("admin", "adminpw"))).json()
        assert rows[0]["session_id"] == "s1"
        lines = (await c.get(f"/admin/api/transcripts/{day}/s1", auth=("admin", "adminpw"))).json()
        assert lines[0]["text"] == "こんにちは"
        r = await c.post("/admin/api/transcripts/s1/hold", auth=("admin", "adminpw"), json={"held": True})
        assert r.json() == {"held": True} and store.holds().keys() == {"s1"}
        assert (await c.get("/admin/api/transcripts/days", auth=("admin", "adminpw"))).json() == [day]
    outside = httpx.ASGITransport(app=create_app(tmp_path / "none", staff, admin), client=("8.8.8.8", 5000))
    async with httpx.AsyncClient(transport=outside, base_url="http://t") as c:
        assert (await c.get("/admin/transcripts/", auth=("admin", "adminpw"))).status_code == 403
    no_pw = AdminPanel(store, AdminConfig(password=""))
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none", None, no_pw), client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/admin/transcripts/", auth=("admin", ""))).status_code == 503


async def test_conversation_turn_info_for_transcripts(tmp_path: Path) -> None:
    c = _conv(SpyLLM([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()]), tmp_path)
    sid = c.session_id
    await _say(c, "国保の切り替えで聞きたいことがあって")
    assert c.turn_info["proposed"] == "refer" and c.session_id == sid
    await _say(c, "いらない")
    assert c.turn_info == {"proposed": None, "reply_to": "refer", "reply_phase": "proposed", "outcome": "declined"}


async def test_dropped_sentences_are_recorded() -> None:
    fac = Facility(name="江田島市役所", location_phrase="1階ロビー")
    d = Dialogue(StubLLM(["少し休んでいってください。", "ロビーにはお茶コーナーがありますので、どうぞ。"]),
                 DialogueConfig(), facility=fac)
    assert await _collect(d, "少し休ませてください") == ["少し休んでいってください。"]
    assert d.last_dropped == [{"reason": "館内の設備・場所の作り話", "text": "ロビーにはお茶コーナーがありますので、どうぞ。"}]


# ------------------------------------------------------------------
# 2) 誤り探しの素材
# ------------------------------------------------------------------


def _export_module() -> Any:
    spec = importlib.util.spec_from_file_location("export_review", REPO_ROOT / "scripts" / "export_review.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_export_review_extracts_four_kinds_and_redacts() -> None:
    ex = _export_module()
    places = ex.place_names("etajima")
    assert "能美町" in places and "江田島" not in places  # 市の名前そのものは伏せない

    def row(t: int, speaker: str, **kw: Any) -> dict[str, Any]:
        return {"ts": f"2026-10-22T10:00:{t:02d}+09:00", "session_id": "s1", "speaker": speaker, **kw}

    rows = [
        row(1, "user", asr_raw="国宝の切り替えで", text="国保の切り替えで"),
        row(2, "assistant", text="保健医療課です。", dropped=[{"reason": "館内の道順", "text": "奥の右です。"}],
            proposal={"proposed": "refer", "reply_to": None, "outcome": None}),
        row(3, "user", asr_raw="いらない", text="いらない"),
        row(4, "assistant", text="わかりました。", proposal={"proposed": None, "reply_to": "refer", "outcome": "declined"}),
        row(5, "user", asr_raw="私は山田です。能美町の中町3丁目に住んでます", text="私は山田です。能美町の中町3丁目に住んでます"),
        row(6, "assistant", text="そうなんですね。"),
        row(7, "user", asr_raw="私は山田です。能美町の中町3丁目に住んでいます", text="私は山田です。能美町の中町3丁目に住んでいます"),
        row(8, "user", asr_raw="090の1234の5678", text="090の1234の5678", ignored="保存直後の断片"),
    ]
    out = ex.extract(rows, places)
    kinds = [x["kind"] for x in out]
    assert kinds == ["asr_fixed", "dropped", "declined", "rephrased"]
    assert out[0]["asr_raw"] == "国宝の切り替えで" and out[0]["text"] == "国保の切り替えで"
    assert out[2]["action"] == "refer" and out[2]["proposed_turn"]["user"] == "国保の切り替えで"
    dump = json.dumps(out, ensure_ascii=False)
    for secret in ("山田", "能美", "中町", "3丁目", "1234"):
        assert secret not in dump
    assert ex.redact("電話は0823-43-1111です", []) == "電話は＊＊です"


# ------------------------------------------------------------------
# 3) アバターの呼び名
# ------------------------------------------------------------------


def test_avatar_name_heads_vocabulary_and_user_dict(tmp_path: Path) -> None:
    words = load_site_words("etajima", SERVER_ROOT / "context", "いどちゃん", 1)
    assert words[0].surface == "いどちゃん" and words[0].reading == "イドチャン" and words[0].accent == 1
    assert words[0].asr
    assert avatar_word("井戸子").reading is None  # 漢字の名前は ASR の語彙だけ
    assert AsrFixes(SERVER_ROOT / "context", tmp_path).apply("伊豆ちゃん、こんにちは") == "いどちゃん、こんにちは"


# ------------------------------------------------------------------
# 4) 紹介状の要約の窓・タグ・案内役のメモ
# ------------------------------------------------------------------


async def test_referral_window_starts_at_trigger_and_procedure_tags_lead(tmp_path: Path) -> None:
    llm = SpyLLM([["それはよかったですね。"], ["国保ですね。<propose:refer>"]],
                 chat_reply=[PHRASED, _refer_json(tags=["雑談", "保険・年金"],
                                                  assistant_note="国宝の切り替えは保健医療課と案内した")])
    c = _conv(llm, tmp_path)
    c._asr_fixes = AsrFixes(SERVER_ROOT / "context", tmp_path)
    await _say(c, "孫が遊びに来てくれて楽しかったんです")
    await _say(c, "国保の切り替えで聞きたいことがあって")
    await _say(c, "お願いします")
    for answer in ["会社を退職したので", "来月からです", "本人です"]:
        await _say(c, answer)
    await c.end_session()
    transcript = next(x for x in llm.chat_inputs if "【来訪者の言葉】" in x)  # 紹介状の構造化に渡した会話
    assert "孫" not in transcript and "それはよかった" not in transcript  # きっかけより前の往復は使わない
    assert "国保の切り替え" in transcript and "会社を退職したので" in transcript  # 聞き取りの答えは入る
    ref = _saved(tmp_path)
    assert ref.tags[0] == "保険・年金" and ref.tags == ["保険・年金", "雑談"]  # 手続きのタグが先、LLM は補助
    assert "国宝" not in ref.assistant_note and "国保" in ref.assistant_note
    assert R["acting"]  # 定型文が読めること（聞き取りの流れは test_actions_unit で見る）


# ------------------------------------------------------------------
# 5) 館内の設備の作り話
# ------------------------------------------------------------------


async def test_invented_building_places_are_not_spoken() -> None:
    fac = Facility(name="江田島市役所", location_phrase="1階ロビー",
                   departments=(Department("保健医療課", "2階"), Department("市民課", "1階")),
                   facilities_in_building=(BuildingFacility("AED", "1階ロビー"),))
    llm = StubLLM(["ロビーにはお茶コーナーがあります。", "2階には相談室があります。", "受付は1階ロビーにあります。",
                   "保健医療課は2階にあります。", "1階ロビーに市民課の窓口があります。", "AEDは1階ロビーにあります。",
                   "公民館で料理教室があります。", "ここには休憩スペースがあります。"])
    d = Dialogue(llm, DialogueConfig(), facility=fac)
    out = await _collect(d, "どんなところですか")
    assert out == ["保健医療課は2階にあります。", "1階ロビーに市民課の窓口があります。", "AEDは1階ロビーにあります。",
                   "公民館で料理教室があります。"]  # 館内の話でない行事の「教室」は残す
    assert [x["reason"] for x in d.last_dropped] == ["館内の設備・場所の作り話"] * 4


# ------------------------------------------------------------------
# 6) 雑談寄りの発話の参考資料
# ------------------------------------------------------------------


class _KB:
    def __init__(self, refs: list[Reference]) -> None:
        self.refs = refs

    def search(self, query: str) -> SearchResult:
        return SearchResult(list(self.refs), 1.0)


async def test_smalltalk_raises_reference_threshold() -> None:
    weak = Reference("やさしい日本語での外国人市民のための防災ガイドブック", "https://x/1", "防災", 0.837, 5.0,
                     chunk_id="c1")
    strong = Reference("国保の加入・脱退時の届出", "https://x/2", "国保", 0.891, 3.0, chunk_id="c2")
    d = Dialogue(StubLLM(["ありがとうございます。"]), DialogueConfig(), knowledge=_KB([weak, strong]))
    await _collect(d, "日本語うまいですね")
    assert d.last_references == [strong] and d.last_search_query == "日本語うまいですね"
    d2 = Dialogue(StubLLM(["保健医療課です。"]), DialogueConfig(), knowledge=_KB([weak, strong]))
    await _collect(d2, "国保の切り替えで聞きたいことがあって")
    assert d2.last_references == [weak, strong]  # 手続きの質問はふだんのしきい値（知識パック側）のまま


async def test_fragment_heard_during_save_is_ignored_even_after_long_readback(tmp_path: Path) -> None:
    from tests.test_actions_unit import FakeClock, _to_ask_phone

    seen: list[bool] = []
    clock = FakeClock()

    class DuringSave(SpyLLM):
        def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> str:
            if "【来訪者の言葉】" in messages[-1]["content"]:
                seen.append(conv.ignores("五六七八。"))  # 紹介状を作っている途中に聞き取った断片
            return super().chat(model, messages, **kw)

    conv = _conv(DuringSave([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()]), tmp_path,
                 clock=clock)
    await _to_ask_phone(conv)
    await _say(conv, "090の1234の5678")
    heard = clock.t + 0.5  # 「はい」の直後に聞いた
    await _say(conv, "はい")
    assert seen == [True]
    clock.t += 14  # 受付番号の読み上げが終わってから順番が回ってきた
    assert conv.ignores("五六七八。", heard_at=heard)  # 聞いた時刻で数える
    assert not conv.ignores("五六七八。")  # 今聞いたものなら 10 秒を過ぎている


async def test_template_turn_does_not_carry_previous_search_state(tmp_path: Path) -> None:
    ref = Reference("国保の加入・脱退時の届出", "https://x/2", "国保", 0.891, 3.0, chunk_id="c2")
    llm = SpyLLM([["国保ですね。[出典:1]<propose:refer>"]], chat_reply=[PHRASED, _refer_json()])
    c = _conv(llm, tmp_path)
    c.dialogue.attach_knowledge(_KB([ref]))
    await _say(c, "国保の切り替えで聞きたいことがあって")
    assert c.dialogue.last_references == [ref] and c.dialogue.last_search_query
    await _say(c, "お願いします")  # 定型文で答えるターン（LLM を通らない）
    d = c.dialogue
    assert (d.last_references, d.last_cited, d.last_search_query, d.last_search_ms, d.last_dropped) == (
        [], [], None, None, [])
