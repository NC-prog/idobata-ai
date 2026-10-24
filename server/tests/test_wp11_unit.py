"""WP11: お知らせ（選ぶ順番・出さない記事・画像を隠す・職員パネルの要約の安全装置・QR）。"""

from __future__ import annotations

import base64
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

from src.config import SERVER_ROOT, DialogueConfig, NewsConfig, StaffConfig
from src.dialogue import Dialogue
from src.knowledge import JST
from src.news import (
    NewsBook,
    NewsRecord,
    load_records,
    load_rules,
    qr_data_url,
    save_record,
    spoken_date,
)
from src.news_panel import NewsPanel, SummaryRejected
from src.referrals import VisitLog
from src.site_context import load_facility
from tests.test_dialogue_unit import StubLLM

FAC = load_facility("etajima", SERVER_ROOT / "context")
RULES = load_rules(NewsConfig().rules_path)
TODAY = date(2026, 10, 24)
NOW = datetime(2026, 10, 24, 10, 0, tzinfo=JST)
BODY = "町道の津浪巡回線で、11月10日から11月14日まで終日全面通行止めを行います。迂回路をご利用ください。"
OK_SUMMARY = "町道の津浪巡回線は、11月10日から11月14日まで終日全面通行止めです。迂回路をご利用ください。"


def _rec(rid: str, **kw: Any) -> NewsRecord:
    base: dict[str, Any] = {"id": rid, "title": f"記事{rid}", "url": f"https://www.city.etajima.hiroshima.jp/{rid}",
                            "published": "2026-10-20", "body": BODY, "relevance": "introduce",
                            "summary": OK_SUMMARY, "summary_checks": {"result": "ok", "reasons": []},
                            "category": "イベント", "expires_at": "2026-11-15", "content_hash": "h"}
    base.update(kw)
    return NewsRecord(**base)


def _book(tmp_path: Path, *records: NewsRecord) -> NewsBook:
    for r in records:
        save_record(tmp_path, r)
    return NewsBook(tmp_path, NewsConfig(), RULES, now=lambda: NOW)


def _checker() -> Dialogue:
    return Dialogue(StubLLM([]), DialogueConfig(), facility=FAC, now=lambda: NOW)


def test_order_priority_then_weight_then_newer_then_less_shown(tmp_path: Path) -> None:
    book = _book(tmp_path,
                 _rec("event_new", published="2026-10-23"),
                 _rec("road_old", category="通行止め", published="2026-10-01"),
                 _rec("facility", category="施設", published="2026-10-22"),
                 _rec("event_shown", published="2026-10-23", shown_count=3),
                 _rec("prio_old", status="priority", published="2026-10-02"),
                 _rec("prio_new", status="priority", published="2026-10-05"))
    assert [r.id for r in book.candidates()] == [
        "prio_new", "prio_old", "road_old", "facility", "event_new", "event_shown"]


def test_stopped_expired_held_and_skip_are_not_candidates(tmp_path: Path) -> None:
    book = _book(tmp_path,
                 _rec("ok"),
                 _rec("stopped", status="stopped"),
                 _rec("expired", expires_at="2026-10-24"),
                 _rec("held", status="held", summary_checks={"result": "held", "reasons": ["英単語"]}),
                 _rec("held_prio", status="priority", summary_checks={"result": "held", "reasons": ["英単語"]}),
                 _rec("skip", relevance="skip"),
                 _rec("nosummary", summary=""))
    assert [r.id for r in book.candidates()] == ["ok"]
    assert book.next_candidate() is not None


def test_image_hidden_is_not_served_or_sent(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"jpeg")
    book = _book(tmp_path, _rec("a", image="a.jpg"), _rec("b", image="a.jpg", image_hidden=True),
                 _rec("c", image="../x.jpg"))
    assert book.image_path("a") == tmp_path / "a.jpg"
    assert book.image_path("b") is None and book.image_path("c") is None
    a, b = book.get("a"), book.get("b")
    assert a is not None and b is not None
    assert book.layout_message(a)["image"] == "/edge/news/a.jpg"
    msg = book.layout_message(b)
    assert msg["image"] is None and msg["type"] == "layout" and msg["mode"] == "news"
    assert msg["qr"].startswith("data:image/png;base64,")


def test_reload_picks_up_changes_after_interval(tmp_path: Path) -> None:
    t = [0.0]
    save_record(tmp_path, _rec("a"))
    book = NewsBook(tmp_path, NewsConfig(reload_sec=60), RULES, clock=lambda: t[0], now=lambda: NOW)
    save_record(tmp_path, _rec("b", status="priority"))
    t[0] = 10
    assert not book.reload()
    t[0] = 61
    assert book.reload() and book.candidates()[0].id == "b"


def test_panel_stop_priority_and_summary_guard(tmp_path: Path) -> None:
    book = _book(tmp_path, _rec("a", published="2026-10-23"), _rec("b", published="2026-10-01"))
    panel = NewsPanel(book, _checker, DialogueConfig(), NewsConfig(), today=lambda: TODAY)
    panel.act("a", "stop")
    assert [r.id for r in book.candidates()] == ["b"]
    panel.act("a", "priority")
    assert book.candidates()[0].id == "a"
    # 要約: 本文に無い日付・英単語・今日などの言い方は保存しない
    for bad, reason in (("町道の津浪巡回線は、11月20日に通行止めです。", "本文に無い数・日付・支払方法"),
                        ("町道の津浪巡回線は、今日から通行止めです。迂回路をご利用ください。", "今日・明日などの言い方（読む日でずれる）"),
                        ("町道の津浪巡回線が road closed です。", "英単語")):
        try:
            panel.edit_summary("a", bad, "staff")
        except SummaryRejected as e:
            assert reason in e.reasons
        else:
            raise AssertionError(bad)
    rec = book.get("a")
    assert rec is not None and rec.summary == OK_SUMMARY and not rec.edited_by_staff
    # 通る文は保存され、職員が直した印が付く（夜の要約は上書きしない）
    held = _rec("h", status="held", summary_checks={"result": "held", "reasons": ["英単語"]})
    book.save(held)
    out = panel.edit_summary("h", OK_SUMMARY, "tanaka")
    assert out["status"] == "auto" and out["edited_by_staff"] and out["edited_by"] == "tanaka"
    assert load_records(tmp_path)[[r.id for r in load_records(tmp_path)].index("h")].summary == OK_SUMMARY
    # 自動に戻しても、要約が通っていなければ保留のまま
    book.save(_rec("h2", status="stopped", summary_checks={"result": "held", "reasons": ["英単語"]}))
    assert panel.act("h2", "auto")["status"] == "held"
    panel.act("a", "hide_image")
    assert book.get("a").image_hidden  # type: ignore[union-attr]


def test_summary_past_dates() -> None:
    import re

    from src.news import summary_problems

    rec = _rec("p", published="2026-09-14", body="令和8年10月1日から乗合割引を開始します。二人以上で乗ると割り引きます。")
    claim, future = re.compile(DialogueConfig().grounding_claim_pattern), re.compile(NewsConfig().summary_future_pattern)

    def check(text: str) -> list[str]:
        return summary_problems(text, rec, _checker(), TODAY, claim, future)

    # 続いている期間の始まりの過ぎた日付は言ってよい
    assert check("10月1日から乗合割引を開始しています。二人以上で乗ると割り引きます。") == []
    # 過ぎた日付をこれからと言う
    assert "過ぎた日付をこれからと言う" in check("10月1日から乗合割引が始まります。")
    # 過ぎた日付だけの文
    assert "過ぎた日付だけの文" in check("10月1日に乗合割引を開始しました。")


async def test_staff_news_routes(tmp_path: Path) -> None:
    from src.edge_server import StaffPanel, create_app
    from src.referrals import ReferralStore, VisitLog

    (tmp_path / "news").mkdir()
    news_dir = tmp_path / "news"
    (news_dir / "a.jpg").write_bytes(b"jpeg")
    book = _book(news_dir, _rec("a", image="a.jpg"), _rec("s", relevance="skip"))
    staff = StaffPanel(ReferralStore(tmp_path, "etajima", 30, csv_export=False), VisitLog(tmp_path, "etajima"),
                       StaffConfig(password="pw"),
                       news=NewsPanel(book, _checker, DialogueConfig(), NewsConfig(), today=lambda: TODAY))
    app = create_app(tmp_path / "none", staff, news=book)
    transport = httpx.ASGITransport(app=app, client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        auth = ("staff", "pw")
        assert (await c.get("/staff/api/news", auth=("staff", "x"))).status_code == 401
        data = (await c.get("/staff/api/news", auth=auth)).json()
        assert [r["id"] for r in data["rows"]] == ["a"] and data["counts"]["skip"] == 1
        assert len((await c.get("/staff/api/news?skip=true", auth=auth)).json()["rows"]) == 2
        assert "お知らせ" in (await c.get("/staff/news/", auth=auth)).text
        bad = await c.put("/staff/api/news/a", auth=auth, json={"summary": "11月30日に通行止めです。"})
        assert bad.status_code == 422 and "本文に無い数・日付・支払方法" in bad.json()["detail"]["reasons"]
        assert (await c.put("/staff/api/news/a", auth=auth, json={"summary": OK_SUMMARY})).status_code == 200
        assert (await c.get("/edge/news/a.jpg")).content == b"jpeg"
        preview = (await c.get("/edge/news/preview/a")).json()
        assert preview["mode"] == "news" and preview["image"] == "/edge/news/a.jpg"
        assert (await c.post("/staff/api/news/a/hide_image", auth=auth)).json()["image_hidden"]
        assert (await c.get("/edge/news/a.jpg")).status_code == 404
        assert (await c.get("/edge/news/preview/a")).json()["image"] is None
        assert (await c.post("/staff/api/news/a/explode", auth=auth)).status_code == 404


def test_spoken_date_and_qr_decodes() -> None:
    cfg = NewsConfig()
    assert spoken_date(TODAY, TODAY, cfg) == "今日"
    assert spoken_date(date(2026, 10, 23), TODAY, cfg) == "昨日"
    assert spoken_date(date(2026, 10, 2), TODAY, cfg) == "10月2日"
    # QR は記事の URL そのまま（読み取りは opencv をテストだけに使う。本体の依存には入れない）
    import cv2
    import numpy as np

    url = "https://www.akiota.jp/soshiki/25/19246.html"
    png = base64.b64decode(qr_data_url(url).split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    got, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert got == url


# ------------------------------------------------------------------
# C: 会話への組み込み（役・提案と同意・紹介の流れ）
# ------------------------------------------------------------------

from tests.test_actions_unit import (
    FAC as ACT_FAC,
)
from tests.test_actions_unit import (
    PHRASED,
    TEMPLATES,
    FakeClock,
    _conv,
    _refer_json,
    _say,
)
from tests.test_actions_unit import StubLLM as ActLLM

N = TEMPLATES["news"]
PROPOSE = N["propose"].format(org="市")


def _news_conv(tmp_path: Path, llm: ActLLM, layouts: list[dict[str, Any]] | None = None,
               book_dir: Path | None = None) -> tuple[Any, NewsBook]:
    news_dir = book_dir or (tmp_path / "news")
    news_dir.mkdir(exist_ok=True)
    book = _book(news_dir, _rec("a", title="津浪巡回線の通行止め", category="通行止め", published="2026-10-21"))
    clock = FakeClock()
    c = _conv(llm, tmp_path, clock=clock)
    c._news, c._news_cfg = book, NewsConfig()

    async def on_layout(msg: dict[str, Any]) -> None:
        if layouts is not None:
            layouts.append(msg)

    c._on_layout = on_layout
    c._clock_for_test = clock
    return c, book


async def _refer_until_saved(c: Any) -> list[str]:
    await _say(c, "国保の切り替えで聞きたいことがあって")
    await _say(c, "お願い")
    for t in ("分からない", "いいから", "言いたくない", "言わなくていいでしょ", "いいです"):
        await _say(c, t)
    return await _say(c, "後日また来ます")


async def test_news_after_referral_saved_then_presented(tmp_path: Path) -> None:
    layouts: list[dict[str, Any]] = []
    llm = ActLLM([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()])
    c, book = _news_conv(tmp_path, llm, layouts)
    out = await _refer_until_saved(c)
    assert out[-1] == PROPOSE and c.pending_action == "news"  # 保存と持ち物の案内の後に提案
    assert c.turn_info["proposed"] == "news"
    c._clock_for_test.t += 11  # 保存直後の断片の無視が明けてから
    shown = await _say(c, "はい")
    assert shown[0] == N["intro"].format(when="今日")  # 会話の日（10/21）が掲載日
    assert "".join(shown[1:]) == OK_SUMMARY
    assert layouts and layouts[0]["mode"] == "news" and layouts[0]["id"] == "a"
    assert c.take_news_closing() == N["closing"].format(org="市") and c.take_news_closing() is None
    assert book.get("a").shown_count == 1  # type: ignore[union-attr]
    # 履歴には要約の本文ではなく定型の一言
    last = c.dialogue.history[-1][1]
    assert "津浪巡回線の通行止め" in last and OK_SUMMARY not in last
    # 1 会話 1 件: もう提案しない
    assert not c._news_available()
    await c.end_session()
    visits = VisitLog(tmp_path, "etajima").read(datetime.now(JST).date())
    news = [p for v in visits for p in v["proposals"] if p["action"] == "news"]
    assert news == [{"action": "news", "outcome": "shown", "trigger": "紹介状の保存の後", "news_id": "a"}]


async def test_news_not_proposed_when_in_hurry(tmp_path: Path) -> None:
    llm = ActLLM([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()])
    c, _ = _news_conv(tmp_path, llm)
    await _say(c, "急いでるんですけど")
    out = await _refer_until_saved(c)
    assert PROPOSE not in out and c.pending_action is None


async def test_news_not_proposed_during_referral_intake(tmp_path: Path) -> None:
    llm = ActLLM([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()])
    c, _ = _news_conv(tmp_path, llm)
    await _say(c, "国保の切り替えで聞きたいことがあって")
    await _say(c, "お願い")
    for t in ("今日は暖かいですね", "いい天気ですね"):  # 聞き取りの途中の雑談（答えとして扱われる）
        assert PROPOSE not in await _say(c, t)
    assert c.pending_action == "refer"


async def test_news_not_proposed_after_crisis(tmp_path: Path) -> None:
    c, _ = _news_conv(tmp_path, ActLLM([["そうですね。"], ["そうですね。"], ["そうですね。"]]))
    await _say(c, "もう死にたいくらいつらいんです")
    for t in ("今日は暖かいですね", "いい天気ですね", "寒くなりましたね"):
        assert PROPOSE not in await _say(c, t)
    assert c.pending_action is None


async def test_news_in_smalltalk_declined_is_not_proposed_again(tmp_path: Path) -> None:
    c, book = _news_conv(tmp_path, ActLLM([["こんにちは。"], ["本当ですね。"], ["お散歩日和ですね。どちらまで行かれましたか？"],
                                            ["そうですね。"], ["そうですね。"]]))
    assert PROPOSE not in await _say(c, "こんにちは")  # 挨拶の直後（会話の 2 ターン目まで）は出さない
    assert PROPOSE not in await _say(c, "今日は暖かいですね")
    out = await _say(c, "いい天気ですね")
    # 雑談が 2 ターン続いた: LLM の問いは落として、提案文だけを問いにする
    assert out == ["お散歩日和ですね。", PROPOSE]
    assert await _say(c, "いいえ") == [N["declined"]]
    for t in ("寒くなりましたね", "いい天気ですね"):
        assert PROPOSE not in await _say(c, t)
    assert book.get("a").shown_count == 0  # type: ignore[union-attr]


async def test_news_after_nothing_else_and_silence_timeout(tmp_path: Path) -> None:
    c, _ = _news_conv(tmp_path, ActLLM([["住民票は市民課です。ほかに何かありますか？"]]))
    await _say(c, "住民票はどこで取れますか")
    assert await _say(c, "いや、ないです") == [PROPOSE]
    assert [s async for s in c.news_timeout()] == [N["declined"]]
    assert c.pending_action is None and not c._news_available()


async def test_news_not_proposed_without_candidates(tmp_path: Path) -> None:
    c, book = _news_conv(tmp_path, ActLLM([["住民票は市民課です。ほかに何かありますか？"]]))
    rec = book.get("a")
    assert rec is not None
    rec.status = "stopped"
    book.save(rec)
    await _say(c, "住民票はどこで取れますか")
    assert PROPOSE not in await _say(c, "いや、ないです")


def test_reception_role_keeps_the_prompt_and_select_role() -> None:
    from src.config import PersonaConfig
    from src.persona import build_system_prompt, load_persona
    from src.roles import PUBLICITY, RECEPTION, Roles, publicity_role, reception_role, select_role

    persona = load_persona(PersonaConfig())
    cfg = DialogueConfig()
    before = build_system_prompt(cfg, FAC, TODAY, [], "いどちゃん", ("lookup",), "こんにちは", persona)
    role = reception_role(persona, FAC.name if FAC else "市役所", ("lookup",))
    after = build_system_prompt(cfg, FAC, TODAY, [], "いどちゃん", ("lookup",), "こんにちは", persona, role.prompt_block)
    assert before == after  # 案内役の決まりを reception に移しただけ（文言は同じ）
    roles = Roles(role, publicity_role())
    assert select_role(roles.state, roles).name == RECEPTION
    roles.state.presenting_news = True
    r, changed = roles.select()
    assert (r.name, r.layout, changed) == (PUBLICITY, "news", True)
    roles.state.presenting_news = False
    assert roles.select() == (roles.reception, True)
    assert ACT_FAC.name


# ------------------------------------------------------------------
# 実機確認の直し（お知らせに届かなかった会話）
# ------------------------------------------------------------------

from tests.test_actions_unit import PROCS
from tests.test_actions_unit import TEMPLATES as ACT_T
from tests.test_dialogue_unit import _collect


class DlgLLM:
    """1 ターンに 1 つずつ応答を返す（ストリームは文ごと）。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> str:
        return ""

    def chat_stream(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        text = self.replies.pop(0) if self.replies else ""
        yield from [x for x in re.split(r"(?<=[。？！])", text) if x]

REFER_PROPOSE = ACT_T["refer"]["propose"]


def _with_procs(c: Any) -> Any:
    c.dialogue._procedures = PROCS  # 承認済みの手続きナレッジ（国保の加入など）を対話に持たせる
    return c


async def test_direct_news_ask_presents_without_proposal(tmp_path: Path) -> None:
    layouts: list[dict[str, Any]] = []
    c, book = _news_conv(tmp_path, ActLLM([["こんにちは。"]]), layouts)
    await _say(c, "こんにちは")
    out = await _say(c, "ところで町からの何か案内ってありますか。")
    assert out[0] == N["direct"] and out[1] == N["intro"].format(when="今日") and "".join(out[2:]) == OK_SUMMARY
    assert layouts[0]["mode"] == "news" and c.take_news_closing() is not None
    assert book.get("a").shown_count == 1  # type: ignore[union-attr]
    # 2 回目は「いま新しいお知らせはありません」
    assert await _say(c, "ほかにお知らせありますか") == [N["none_direct"]]


async def test_direct_news_ask_closes_pending_proposal(tmp_path: Path) -> None:
    c, _ = _news_conv(tmp_path, ActLLM([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()]))
    assert (await _say(c, "国保の切り替えで聞きたいことがあって"))[-1] == REFER_PROPOSE
    out = await _say(c, "町からのお知らせってある？")
    assert out[0] == N["direct"] and c.pending_action is None
    assert [p["outcome"] for p in c._session.proposals] == ["declined", "shown"]


async def test_direct_news_ask_with_no_candidates(tmp_path: Path) -> None:
    c, book = _news_conv(tmp_path, ActLLM([]))
    rec = book.get("a")
    assert rec is not None
    rec.status = "stopped"
    book.save(rec)
    assert await _say(c, "最近なにかイベントありますか") == [N["none_direct"]]


def test_news_ask_pattern() -> None:
    import yaml

    from src.config import ReferralConfig

    t = yaml.safe_load((SERVER_ROOT / "context" / "actions.yaml").read_text(encoding="utf-8"))["actions"]["news"]
    ask, req = re.compile(t["ask_pattern"]), re.compile(t["ask_request_pattern"])
    proc = re.compile(ReferralConfig().procedure_pattern)

    def hit(x: str) -> bool:
        return bool(ask.search(x) and req.search(x) and not proc.search(x))

    for x in ("ところで町からの何か案内ってありますか。", "市から何かお知らせある？", "最近なにかイベントありますか",
              "お知らせを教えて", "催しはありますか"):
        assert hit(x), x
    for x in ("国保の手続きの案内はありますか", "お知らせありがとう", "今日は暖かいですね", "住民票の届出の案内は"):
        assert not hit(x), x


async def test_rule_refer_after_two_turns_on_approved_procedure(tmp_path: Path) -> None:
    c, _ = _news_conv(tmp_path, ActLLM([["国保ですね。保健医療課です。"], ["退職のときは保健医療課でご案内します。"]]))
    _with_procs(c)
    assert REFER_PROPOSE not in await _say(c, "うんと国保のことで聞きに来ました。")
    out = await _say(c, "いや、えっと退職とかで。")
    assert out[-1] == REFER_PROPOSE and c.pending_action == "refer"  # LLM が 2 ターン提案しなかった: 規則で


async def test_news_after_closing_reply_on_procedure_topic(tmp_path: Path) -> None:
    c, _ = _news_conv(tmp_path, ActLLM([["国保ですね。<propose:refer>"]]))
    _with_procs(c)
    await _say(c, "国保のことで聞きに来ました。")  # 紹介状の提案（LLM）
    await _say(c, "退職とかで。")  # 返事として分からない → 聞き直し（話題の続きに数える）
    await _say(c, "いえ、大丈夫です。")  # 辞退
    assert await _say(c, "わかりました。") == [PROPOSE]  # 手続きの話の後の締めの返事 → お知らせの提案
    assert c._session.proposals[-1]["trigger"] == "手続きの話の後の締めの返事"


async def test_smalltalk_counts_replies_and_needs_three_turns(tmp_path: Path) -> None:
    c, _ = _news_conv(tmp_path, ActLLM([["お変わりありませんか？"], ["お住まいの近くはもう紅葉していますか？"],
                                        ["そうなんですね。"]]))
    assert PROPOSE not in await _say(c, "こんにちは。")
    assert PROPOSE not in await _say(c, "うんないです。")  # 会話の 2 ターン目までは出さない
    out = await _say(c, "うんまだ知ってないです。")  # 返事も雑談に数える（天気などの語が無くても）
    assert out[-1] == PROPOSE


async def test_offer_once_per_conversation_even_as_single_sentence() -> None:
    d = Dialogue(DlgLLM(["お変わりありませんか？", "お変わりありませんか？", "最近どうですか？", "お元気でしたか？"]),
                 DialogueConfig(), now=lambda: NOW)
    assert await _collect(d, "こんにちは") == ["お変わりありませんか？"]
    # 1 文だけの応答も安全装置に通す（前は最後の 1 文が素通りした）。類語も同じ 1 回に数える
    assert await _collect(d, "はい") == ["そうですか。"]
    assert await _collect(d, "ないですよ") == [DialogueConfig().empty_negative_filler]
    assert await _collect(d, "元気ですか？") == [DialogueConfig().empty_question_filler]


async def test_agree_phrase_only_after_opinion() -> None:
    d = Dialogue(DlgLLM(["おっしゃる通りですね。", "おっしゃる通り、退職のことですね。", "おっしゃる通りですね。"]),
                 DialogueConfig(), now=lambda: NOW)
    assert await _collect(d, "うんまだ知ってないです。") == ["そうですか。"]
    assert await _collect(d, "退職とかで") == ["退職のことですね。"]
    assert await _collect(d, "役所はもっと早く開けるべきだと思うんよ") == ["おっしゃる通りですね。"]


async def test_invented_department_business_is_dropped() -> None:
    fac = load_facility("akiota", SERVER_ROOT / "context")
    d = Dialogue(DlgLLM(["町からの案内やお知らせは、地域協働課で配布しています。お気軽にどうぞ。"]),
                 DialogueConfig(), facility=fac, now=lambda: NOW)
    out = await _collect(d, "町からの何か案内ってありますか")
    assert all("地域協働課" not in x for x in out)
    assert any(x["reason"] == "課の業務の作り話" for x in d.last_dropped)
    # 課の handles にある業務は言ってよい（空き家バンクは地域協働課）
    d2 = Dialogue(DlgLLM(["空き家バンクは地域協働課で受け付けています。"]), DialogueConfig(), facility=fac,
                  now=lambda: NOW)
    assert await _collect(d2, "空き家バンクに登録したい") == ["空き家バンクは地域協働課で受け付けています。"]


async def test_first_sentence_uses_procedure_name() -> None:
    fac = load_facility("akiota", SERVER_ROOT / "context")
    d = Dialogue(DlgLLM(["国保税についてご相談ですね。窓口でご案内します。"]), DialogueConfig(), facility=fac,
                 procedures=PROCS, now=lambda: NOW)
    out = await _collect(d, "うんと国保のことで聞きに来ました。")
    assert out[0].startswith("国民健康保険の加入（退職・転入など）についてご相談ですね") and "国保税" not in out[0]
