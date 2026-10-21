"""WP04: 相槌・同意付きの行為（調べる・つなぐ）・紹介状・来訪記録・職員パネルのテスト。"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.actions import Conversation, classify_reply, load_templates
from src.backchannel import BackchannelBank, strip_leading_backchannel
from src.config import SERVER_ROOT, ActionsConfig, DialogueConfig, ReferralConfig, StaffConfig
from src.dialogue import Dialogue
from src.knowledge import JST, Reference
from src.lookup import LookupResult, candidate_links, chunk_plain, score_label
from src.procedures import load_procedures
from src.referrals import (
    ReferralStore,
    TidbitStore,
    VisitLog,
    find_phone,
    phone_for_speech,
    pick_tags,
    today_stats,
)
from src.site_context import Department, Facility

TAGS = ["子育て", "介護", "保険・年金", "観光", "行事", "雑談"]
FAC = Facility(name="江田島市役所", location_phrase="1階ロビー",
               departments=(Department("市民課", "1階", handles=("住民票", "戸籍")),
                            Department("税務課", "1階", handles=("国民健康保険税", "税証明書")),
                            Department("保健医療課", "2階", handles=("国民健康保険（資格・給付）", "後期高齢者医療")),
                            Department("高齢介護課", "2階", handles=("介護保険",))))
PROCS = load_procedures(SERVER_ROOT / "context")


# ------------------------------------------------------------------
# 返事の判定・相槌
# ------------------------------------------------------------------

WORDS, TEMPLATES = load_templates(SERVER_ROOT / "context" / "actions.yaml")


@pytest.mark.parametrize(("text", "verdict"), [
    ("うん", "yes"), ("はい", "yes"), ("お願いします", "yes"), ("ええ、頼みます", "yes"),
    ("いいですよ", "yes"), ("いや、お願いします", "yes"),
    ("いらない", "no"), ("大丈夫です", "no"), ("はい、大丈夫です", "no"), ("結構です", "no"),
    ("いいです", "no"), ("いいえ", "no"),
    ("ところで駐車場は", "unclear"), ("えっと", "unclear"),
])
def test_classify_reply(text: str, verdict: str) -> None:
    assert classify_reply(text, WORDS) == verdict


def test_backchannel_rules_rotate_and_skip_short_replies() -> None:
    bank = BackchannelBank(SERVER_ROOT / "context" / "backchannel.yaml")
    assert bank.choose("うん") is None
    assert bank.choose("はい。") is None
    first = bank.choose("住民票はどこで取れますか")
    second = bank.choose("いつ開いてますか")
    assert first is not None and second is not None and first[0] != second[0]  # 同じ相槌を続けない
    assert bank.choose("母が亡くなって、つらくて") == ("ええ", 0.6)  # 打ち明けは間を長く
    assert bank.choose("今日は寒いですね")[0] in ("ええ", "ほんとに")  # type: ignore[index]


def test_strip_leading_backchannel() -> None:
    assert strip_leading_backchannel("はい、市民課は1階です。", "はい") == "市民課は1階です。"
    assert strip_leading_backchannel("はい。", "はい") == ""
    assert strip_leading_backchannel("ええ、そうです。", "はい") == "ええ、そうです。"


# ------------------------------------------------------------------
# 会話（提案 → 返事 → 実行）
# ------------------------------------------------------------------


class StubLLM:
    """chat_stream は決まった応答（文の配列）を順に返す。chat は構造化・検索語用。"""

    def __init__(self, replies: list[list[str]], chat_reply: str | list[str] = "{}") -> None:
        self.replies = replies
        self.chat_reply = chat_reply
        self.stream_calls: list[list[dict[str, Any]]] = []

    def chat_stream(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Iterator[str]:
        self.stream_calls.append(messages)
        yield from (self.replies.pop(0) if self.replies else ["そうなんですね。"])

    def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> str:
        if isinstance(self.chat_reply, list):
            return self.chat_reply.pop(0) if self.chat_reply else "{}"
        return self.chat_reply


class StubLookup:
    def __init__(self, result: LookupResult) -> None:
        self.result = result
        self.sources = [object()]
        self.asked: list[str] = []

    async def run(self, user_text: str) -> LookupResult:
        self.asked.append(user_text)
        return self.result


def _conv(llm: StubLLM, tmp_path: Path, lookup: Any = None, enabled: tuple[str, ...] = ("lookup", "refer"),
          states: list[str] | None = None, tidbits: TidbitStore | None = None) -> Conversation:
    d = Dialogue(llm, DialogueConfig(), facility=FAC, enabled_actions=enabled)

    async def on_state(v: str) -> None:
        if states is not None:
            states.append(v)

    return Conversation(
        d, llm, "/models/llm", ActionsConfig(enabled=enabled), ReferralConfig(), FAC, TAGS,
        lookup=lookup, referrals=ReferralStore(tmp_path, "etajima", 30, csv_export=True),
        visits=VisitLog(tmp_path, "etajima"), on_state=on_state, edge_id=lambda: "edge-1",
        tidbits=tidbits, procedures=PROCS,
    )


async def _say(c: Conversation, text: str) -> list[str]:
    return [s async for s in c.respond(text)]


async def test_lookup_proposal_accepted_and_answered(tmp_path: Path) -> None:
    ref = Reference("道の駅の秋のイベント", "https://example.jp/1", "10月25日に新米まつり。", 0.0, 3.0,
                    "semi", "道の駅のお知らせ")
    lookup = StubLookup(LookupResult(found=True, references=[ref]))
    llm = StubLLM([["今週末の情報は手元にありません。<propose:lookup>"], ["25日に新米まつりがあります。[出典:1]"]])
    states: list[str] = []
    c = _conv(llm, tmp_path, lookup, states=states)
    first = await _say(c, "今週末、道の駅で何かやってますか")
    assert first == ["今週末の情報は手元にありません。", TEMPLATES["lookup"]["propose"]]
    assert c.phase == "proposed"
    second = await _say(c, "うん")
    assert second[0] == TEMPLATES["lookup"]["acting"]
    assert "新米まつり" in second[1]
    assert second[-1] == "詳しくは道の駅のお知らせにも載っています。"
    assert lookup.asked == ["今週末、道の駅で何かやってますか"]
    assert states == ["thinking"] and c.phase == "listen"
    # 調べた結果は、元の質問に参考資料として渡して答える
    assert "新米まつり" in llm.stream_calls[-1][-1]["content"]


async def test_lookup_not_found_is_queued_message(tmp_path: Path) -> None:
    lookup = StubLookup(LookupResult(found=False, queued=True))
    c = _conv(StubLLM([["分かりません。<propose:lookup>"]]), tmp_path, lookup)
    await _say(c, "道の駅のイベントは")
    out = await _say(c, "お願い")
    assert out == [TEMPLATES["lookup"]["acting"], TEMPLATES["lookup"]["not_found"]]


async def test_decline_returns_to_normal_dialogue(tmp_path: Path) -> None:
    llm = StubLLM([["分かりません。<propose:lookup>"], ["ほかにも聞いてくださいね。"]])
    c = _conv(llm, tmp_path, StubLookup(LookupResult(found=False)))
    await _say(c, "道の駅のイベントは")
    assert await _say(c, "いらない") == [TEMPLATES["lookup"]["declined"]]
    assert c.phase == "listen"
    assert await _say(c, "ありがとう") == ["ほかにも聞いてくださいね。"]  # 通常の対話に戻る


async def test_unclear_reply_asks_again_once_then_declines(tmp_path: Path) -> None:
    c = _conv(StubLLM([["分かりません。<propose:lookup>"]]), tmp_path, StubLookup(LookupResult(found=False)))
    await _say(c, "道の駅のイベントは")
    assert await _say(c, "えっと") == [TEMPLATES["lookup"]["reask"]]
    assert await _say(c, "うーん") == [TEMPLATES["lookup"]["declined"]]
    assert c.phase == "listen"


async def test_proposal_ignored_when_action_unavailable(tmp_path: Path) -> None:
    c = _conv(StubLLM([["分かりません。<propose:lookup>"]]), tmp_path, lookup=None)
    assert await _say(c, "道の駅のイベントは") == ["分かりません。"]
    assert c.phase == "listen"


def _refer_json(**kw: Any) -> str:
    base = {"department": "保健医療課", "category": "手続き", "summary": "国保の切り替えについて聞きたい。",
            "urgency": "通常", "contact_pref": "窓口で待つ", "name": None, "phone": None,
            "needs_action": True, "tags": ["保険・年金", "勝手なタグ"], "topic_free": "国保の切り替え"}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


async def test_refer_without_name_saves_and_reads_receipt(tmp_path: Path) -> None:
    llm = StubLLM([["保健医療課の担当です。<propose:refer>"]], chat_reply=_refer_json(name="山田"))
    c = _conv(llm, tmp_path)
    await _say(c, "国保の切り替えで聞きたいことがあって")
    out = await _say(c, "お願い")
    assert out[0] == TEMPLATES["refer"]["acting"]
    assert "お名前はおっしゃらなくても大丈夫です" in out[1]  # 氏名を言っていないので聞かない
    store = ReferralStore(tmp_path, "etajima", 30, csv_export=True)
    (ref,) = store.list(datetime.now(JST).date())
    assert ref.receipt in out[1] and c.last_receipt == ref.receipt
    assert ref.name is None  # LLM が名前を作っても、来訪者が言っていなければ残さない
    assert ref.phone is None and ref.department == "保健医療課"
    assert ref.tags == ["保険・年金"]  # 統制語彙に無いタグは捨てる
    assert ref.needs_action is True and ref.topic_free == "国保の切り替え" and ref.edge_id == "edge-1"


async def test_lookup_on_procedure_is_replaced_by_refer(tmp_path: Path) -> None:
    llm = StubLLM([["国保のことですね。<propose:lookup>"]], chat_reply=_refer_json())
    c = _conv(llm, tmp_path, StubLookup(LookupResult(found=False)))
    out = await _say(c, "国保の切り替えで聞きたいことがあって")
    assert out[-1] == TEMPLATES["refer"]["propose"]  # 手続きの相談は調べるより、つなぐ


async def test_refer_on_event_question_is_replaced_by_lookup(tmp_path: Path) -> None:
    c = _conv(StubLLM([["商工観光課へどうぞ。<propose:refer>"]]), tmp_path, StubLookup(LookupResult(found=False)))
    out = await _say(c, "今週末は道の駅で何かやってますか")
    assert out[-1] == TEMPLATES["lookup"]["propose"]


async def test_missing_tag_is_filled_by_rules(tmp_path: Path) -> None:
    llm = StubLLM([["手元の資料にありませんので、商工観光課へお尋ねください。"], ["住民票は市民課です。"],
                   ["保健医療課でご案内します。"]])
    c = _conv(llm, tmp_path, StubLookup(LookupResult(found=False)))
    out = await _say(c, "今週末、道の駅で何かやってますか")
    assert out[-1] == TEMPLATES["lookup"]["propose"]
    await _say(c, "いらない")
    assert await _say(c, "住民票はどこで取れますか") == ["住民票は市民課です。"]  # 場所を聞いただけなら提案しない
    out = await _say(c, "国保の切り替えで聞きたいことがあって")
    assert out[-1] == TEMPLATES["refer"]["propose"]


async def test_lookup_on_smalltalk_is_dropped_and_new_topic_skips_reask(tmp_path: Path) -> None:
    llm = StubLLM([["そうですね。<propose:lookup>"], ["分かりません。<propose:lookup>"], ["ようこそ。"]])
    c = _conv(llm, tmp_path, StubLookup(LookupResult(found=False)))
    assert await _say(c, "最近、急に寒くなりましたね。") == ["そうですね。"]  # 尋ねていないので提案しない
    await _say(c, "道の駅のイベントはありますか")
    assert c.phase == "proposed"
    # 返事ではなく別の話: 聞き直さず、その話に答える
    assert await _say(c, "こんにちは。ちょっと寄ってみました。") == ["ようこそ。"]
    assert c.phase == "listen"


async def test_short_ack_after_action_gets_fixed_reply(tmp_path: Path) -> None:
    llm = StubLLM([["保健医療課です。<propose:refer>"], ["担当者がそろいましたので、お呼びしますね。"]],
                  chat_reply=_refer_json())
    c = _conv(llm, tmp_path)
    await _say(c, "国保の切り替えで聞きたいことがあって")
    await _say(c, "お願い")
    assert await _say(c, "うん") == [TEMPLATES["common"]["ack"]]  # LLM に続けさせない
    assert len(llm.stream_calls) == 1


async def test_refer_on_seasonal_smalltalk_becomes_share(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    c = _conv(StubLLM([["いいですね。<propose:refer>"]]), tmp_path, enabled=SHARE, tidbits=store)
    out = await _say(c, "庭の柿がたくさん実ったんですよ")
    assert out[-1] == TEMPLATES["share"]["propose"]


async def test_refer_with_phone_is_read_back(tmp_path: Path) -> None:
    llm = StubLLM([["保健医療課です。", "ご連絡は 090-1234-5678 ですね。<propose:refer>"]],
                  chat_reply=_refer_json(name="田中"))
    c = _conv(llm, tmp_path)
    await _say(c, "田中です。国保のことで、電話は090-1234-5678です")
    out = await _say(c, "お願いします")
    assert out[-1] == TEMPLATES["refer"]["confirm_phone"].format(phone=phone_for_speech("090-1234-5678"))
    assert c.phase == "confirm_phone"
    done = await _say(c, "はい")
    store = ReferralStore(tmp_path, "etajima", 30, csv_export=True)
    (ref,) = store.list(datetime.now(JST).date())
    assert ref.phone == "090-1234-5678" and ref.name == "田中"
    assert "お名前は" not in done[0] and ref.receipt in done[0]


async def test_refer_phone_rejected_is_dropped(tmp_path: Path) -> None:
    llm = StubLLM([["わかりました。<propose:refer>"]], chat_reply=_refer_json())
    c = _conv(llm, tmp_path)
    await _say(c, "国保のことで、電話は０９０－１２３４－５６７８です")
    await _say(c, "お願い")
    out = await _say(c, "違います")
    assert out[0] == TEMPLATES["refer"]["phone_wrong"] or out[0].startswith(TEMPLATES["refer"]["phone_wrong"][:6])
    (ref,) = ReferralStore(tmp_path, "etajima", 30, csv_export=True).list(datetime.now(JST).date())
    assert ref.phone is None


async def test_end_session_writes_anonymous_visit(tmp_path: Path) -> None:
    llm = StubLLM([["そうなんですね。"]], chat_reply=json.dumps({"tags": ["雑談", "行事", "勝手"]}))
    c = _conv(llm, tmp_path)
    c.note_backchannel()  # 最初の相槌は応答より先に出る（そこで会話が始まる）
    await _say(c, "今日は秋祭りの準備なんです")
    c.note_backchannel()
    await c.end_session()
    (line,) = (tmp_path / "visits" / "etajima").glob("*.jsonl")
    rec = json.loads(line.read_text(encoding="utf-8"))
    assert rec["turns"] == 1 and rec["backchannel_count"] == 2 and rec["tags"] == ["雑談", "行事"]
    assert "秋祭り" not in json.dumps(rec, ensure_ascii=False)  # 本文は残さない
    assert rec["edge_id"] == "edge-1" and rec["referral_id"] is None
    assert c.dialogue.history_len == 0


# ------------------------------------------------------------------
# 紹介状の保存・期限・CSV、集計
# ------------------------------------------------------------------


def _fields(**kw: Any) -> dict[str, Any]:
    f = {"department": "市民課", "category": "手続き", "summary": "住民票の写しが欲しい。", "urgency": "通常",
         "contact_pref": "窓口で待つ", "name": None, "phone": None, "needs_action": True,
         "tags": ["住民票・戸籍"], "topic_free": "住民票"}
    f.update(kw)
    return f


def test_store_receipts_update_and_csv(tmp_path: Path) -> None:
    store = ReferralStore(tmp_path, "akiota", 30, csv_export=True)
    a = store.create(_fields())
    b = store.create(_fields(urgency="急ぐ"))
    assert a.receipt != b.receipt and len(a.receipt) == 4 and a.receipt.isdigit()
    today = date.fromisoformat(a.created_at[:10])
    upd = store.update(a.id, today, status="完了", handled_by_role="窓口", resolution="完了")
    assert upd is not None and upd.status == "完了" and upd.handled_at and upd.handled_by_role == "窓口"
    assert store.update(a.id, today, status="勝手な状態") is not None  # 知らない状態は無視
    raw = (tmp_path / "referrals" / "akiota" / f"{today.isoformat()}.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM（kintone 取り込み用）
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    assert rows[0][0] == "受付番号" and len(rows) == 3


def test_retention_zero_removes_yesterday(tmp_path: Path) -> None:
    store = ReferralStore(tmp_path, "etajima", 0, csv_export=False)
    ref = store.create(_fields())
    today = date.fromisoformat(ref.created_at[:10])
    yday = today - timedelta(days=1)
    old = tmp_path / "referrals" / "etajima" / yday.isoformat()
    old.mkdir(parents=True)
    (old / "x.json").write_text("{}", encoding="utf-8")
    assert store.cleanup(today) == 1
    assert not old.exists() and store.list(today)  # 今日の分は残る


def test_today_stats(tmp_path: Path) -> None:
    store = ReferralStore(tmp_path, "etajima", 30, csv_export=False)
    visits = VisitLog(tmp_path, "etajima")
    ref = store.create(_fields(tags=["介護"]))
    d = date.fromisoformat(ref.created_at[:10])
    visits.write({"started_at": ref.created_at, "language": "ja", "tags": ["介護", "雑談"], "referral_id": ref.id})
    visits.write({"started_at": ref.created_at, "language": "en", "tags": ["観光"], "referral_id": None})
    st = today_stats(store, visits, d)
    assert st["visits"] == 2 and st["referrals"] == 1 and st["pending_action"] == 1
    assert st["languages"] == {"ja": 1, "en": 1} and ("介護", 1) in st["top_tags"]


def test_phone_and_tags_helpers() -> None:
    assert find_phone("電話は０８２３－４３－１１１１です") == "0823-43-1111"
    assert find_phone("電話はしません") is None
    assert phone_for_speech("090-1234-5678") == "0 9 0、1 2 3 4、5 6 7 8"
    assert pick_tags(["介護", "介護", "なし", "観光"], TAGS) == ["介護", "観光"]


# ------------------------------------------------------------------
# 職員パネル
# ------------------------------------------------------------------


async def test_staff_panel_auth_lan_and_update(tmp_path: Path) -> None:
    from src.edge_server import StaffPanel, create_app

    store = ReferralStore(tmp_path, "etajima", 30, csv_export=False)
    ref = store.create(_fields())
    panel = StaffPanel(store, VisitLog(tmp_path, "etajima"), StaffConfig(password="pw"))
    app = create_app(tmp_path / "none", panel)
    transport = httpx.ASGITransport(app=app, client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/staff/api/referrals")).status_code == 401
        assert (await c.get("/staff/api/referrals", auth=("staff", "x"))).status_code == 401
        rows = (await c.get("/staff/api/referrals", auth=("staff", "pw"))).json()
        assert rows[0]["receipt"] == ref.receipt
        r = await c.post(f"/staff/api/referrals/{ref.created_at[:10]}/{ref.id}", auth=("staff", "pw"),
                         json={"status": "対応中", "needs_action": False})
        assert r.json()["status"] == "対応中" and r.json()["needs_action"] is False
        page = await c.get("/staff/", auth=("staff", "pw"))
        assert page.status_code == 200 and "職員パネル" in page.text
        assert (await c.get("/staff/api/stats", auth=("staff", "pw"))).json()["referrals"] == 1
    outside = httpx.ASGITransport(app=app, client=("8.8.8.8", 5000))
    async with httpx.AsyncClient(transport=outside, base_url="http://t") as c:
        assert (await c.get("/staff/api/referrals", auth=("staff", "pw"))).status_code == 403


async def test_staff_panel_disabled_without_password(tmp_path: Path) -> None:
    from src.edge_server import StaffPanel, create_app

    panel = StaffPanel(ReferralStore(tmp_path, "e", 30, False), VisitLog(tmp_path, "e"), StaffConfig())
    transport = httpx.ASGITransport(app=create_app(tmp_path, panel), client=("127.0.0.1", 1))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/staff/api/referrals", auth=("staff", ""))).status_code == 503


# ------------------------------------------------------------------
# 調べる（取得先の解析）
# ------------------------------------------------------------------


def test_lookup_link_extraction_and_scoring() -> None:
    atom = ('<feed><entry><title>道の駅で新米まつり</title><link href="https://www.akiota.jp/soshiki/8/1.html"/>'
            '</entry><entry><title>入札のお知らせ</title><link href="https://other.example/2"/></entry></feed>')
    links = candidate_links(atom, "https://www.akiota.jp/rss/10/list1.xml", ["www.akiota.jp"])
    assert links == [("https://www.akiota.jp/soshiki/8/1.html", "道の駅で新米まつり")]
    assert score_label("道の駅で新米まつり", "道の駅 まつり") > score_label("入札のお知らせ", "道の駅 まつり")
    assert all(len(c) <= 700 for c in chunk_plain("あいうえお。" * 300))


def test_safe_query_blocks_english_and_banned() -> None:
    d = Dialogue(StubLLM([]), DialogueConfig())
    assert d.safe_query("道の駅 イベント\n") == "道の駅 イベント"
    assert d.safe_query("michi no eki") is None
    assert d.safe_query("") is None


async def test_gather_within_keeps_finished_and_cancels_slow() -> None:
    import asyncio

    from src.lookup import _gather_within

    async def fast() -> str:
        return "ok"

    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    out = await _gather_within([fast(), slow()], 0.1)
    assert out[0] == "ok" and isinstance(out[1], TimeoutError)



# ------------------------------------------------------------------
# 覚えておく（今日の小ネタ。WP04 C）
# ------------------------------------------------------------------

SHARE = ("lookup", "refer", "share")


def _tid(text: str) -> str:
    return json.dumps({"text": text}, ensure_ascii=False)


async def test_share_rewrites_confirms_and_saves(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    llm = StubLLM([["いいですね。<propose:share>"]], chat_reply=[_tid("今朝は大柿の港で牡蠣の水揚げがにぎやかだったそうです")])
    c = _conv(llm, tmp_path, enabled=SHARE, tidbits=store)
    out = await _say(c, "今朝、大柿の港で牡蠣の水揚げがすごかったよ")
    assert out[-1] == TEMPLATES["share"]["propose"]
    confirm = await _say(c, "うん")
    assert confirm == [TEMPLATES["share"]["confirm_text"].format(text="今朝は大柿の港で牡蠣の水揚げがにぎやかだったそうです")]
    assert c.phase == "confirm_share" and store.active() == []  # 確かめるまでは保存しない
    assert await _say(c, "はい") == [TEMPLATES["share"]["done"]]
    (item,) = store.active()
    assert item["text"].startswith("今朝は大柿") and c.phase == "listen"
    # 同じ話をもう一度しても、二重には預からない
    c2 = _conv(StubLLM([["いいですね。<propose:share>"]]), tmp_path, enabled=SHARE, tidbits=store)
    assert await _say(c2, "今朝は大柿の港で牡蠣の水揚げがにぎやかだったよ") == ["いいですね。"]


async def test_share_rejected_at_confirmation_is_not_saved(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    llm = StubLLM([["いいですね。<propose:share>"]], chat_reply=[_tid("今朝は大柿の港で牡蠣の水揚げがにぎやかだったそうです")])
    c = _conv(llm, tmp_path, enabled=SHARE, tidbits=store)
    await _say(c, "今朝、大柿の港で牡蠣の水揚げがすごかったよ")
    await _say(c, "うん")
    assert await _say(c, "やっぱりやめて") == [TEMPLATES["share"]["declined"]]
    assert store.active() == []


async def test_share_rewrite_with_personal_details_is_dropped(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    llm = StubLLM([["いいですね。<propose:share>"]], chat_reply=[_tid("山田さんの息子が港で牡蠣をたくさん獲ったそうです")])
    c = _conv(llm, tmp_path, enabled=SHARE, tidbits=store)
    await _say(c, "うちの息子が牡蠣をたくさん獲ってきたよ")
    assert await _say(c, "うん") == [TEMPLATES["share"]["failed"]]
    assert store.active() == [] and c.phase == "listen"


async def test_tidbit_offered_once_per_conversation_and_not_to_its_author(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    item = store.add("今朝は大柿の港で牡蠣の水揚げがにぎやかだったそうです")
    llm = StubLLM([["さっき来た方が、大柿で牡蠣の水揚げがにぎやかだったって言ってましたよ。"], ["そうですね。"],
                   ["そうですね。"]])
    c = _conv(llm, tmp_path, enabled=SHARE, tidbits=store)
    await _say(c, "今日はいい天気ですね")
    assert "さっき来た方の話" in llm.stream_calls[0][-1]["content"]
    await _say(c, "ほんとに気持ちいいですね")
    assert "さっき来た方の話" not in llm.stream_calls[1][-1]["content"]  # 1 会話に 1 つまで
    # 預けた本人の会話では、その小ネタを使わない
    llm3 = StubLLM([["そうですね。"]])
    c3 = _conv(llm3, tmp_path, enabled=SHARE, tidbits=store)
    c3._ensure_session("ja")
    assert c3._session is not None
    c3._session.own_tidbits.append(item["id"])
    await _say(c3, "今日はいい天気ですね")
    assert "さっき来た方の話" not in llm3.stream_calls[0][-1]["content"]


async def test_tidbit_not_offered_on_questions(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    store.add("今朝は大柿の港で牡蠣の水揚げがにぎやかだったそうです")
    llm = StubLLM([["市民課です。"]])
    c = _conv(llm, tmp_path, enabled=SHARE, tidbits=store)
    await _say(c, "住民票はどこで取れますか")
    assert "さっき来た方の話" not in llm.stream_calls[0][-1]["content"]


def test_tidbit_retention_and_delete(tmp_path: Path) -> None:
    store = TidbitStore(tmp_path, "etajima", 7)
    old = {"id": "old1", "created_at": (datetime.now(JST) - timedelta(days=8)).isoformat(), "text": "古い話です"}
    store.path.parent.mkdir(parents=True)
    store.path.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")
    new = store.add("新しい話です")
    assert [t["id"] for t in store.active()] == [new["id"]]
    assert store.cleanup() == 1
    assert store.delete(new["id"]) and store.active() == [] and not store.delete("nope")


async def test_staff_panel_lists_and_deletes_tidbits(tmp_path: Path) -> None:
    from src.edge_server import StaffPanel, create_app

    tidbits = TidbitStore(tmp_path, "etajima", 7)
    item = tidbits.add("今朝は大柿の港で牡蠣の水揚げがにぎやかだったそうです")
    panel = StaffPanel(ReferralStore(tmp_path, "etajima", 30, csv_export=False), VisitLog(tmp_path, "etajima"),
                       StaffConfig(password="pw"), tidbits)
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none", panel), client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/staff/api/tidbits")).status_code == 401
        rows = (await c.get("/staff/api/tidbits", auth=("staff", "pw"))).json()
        assert rows[0]["id"] == item["id"]
        assert (await c.delete(f"/staff/api/tidbits/{item['id']}", auth=("staff", "pw"))).status_code == 200
        assert (await c.get("/staff/api/tidbits", auth=("staff", "pw"))).json() == []
        assert (await c.delete("/staff/api/tidbits/nope", auth=("staff", "pw"))).status_code == 404


# ------------------------------------------------------------------
# 担当課は規則で決める・紹介状の範囲（WP04 追加）
# ------------------------------------------------------------------


def test_departments_by_rules_with_aliases() -> None:
    from src.site_context import load_facility

    fac = load_facility("etajima", SERVER_ROOT / "context")
    assert PROCS.departments_for("国保の切り替えで聞きたいことがあって", fac) == ["保健医療課"]
    assert PROCS.departments_for("国保税の納付が遅れそうで", fac) == ["税務課"]  # 長い語を優先
    assert PROCS.departments_for("後期高齢の保険証のことで", fac) == ["保健医療課"]
    assert PROCS.departments_for("要介護認定を受けたい", fac) == ["高齢介護課"]
    assert PROCS.departments_for("印鑑証明が欲しい", fac) == ["市民課"]
    assert PROCS.departments_for("粗大ごみの出し方", fac) == ["生活環境課"]
    assert PROCS.departments_for("ちょっと困ったことがあって", fac) == []
    assert len(PROCS.departments_for("住民票と介護保険のことで", fac)) == 2


def test_procedure_words_for_asr_and_voicevox() -> None:
    from src.asr_client import load_site_vocabulary
    from src.site_context import load_site_words

    vocab = load_site_vocabulary("etajima", SERVER_ROOT / "context")
    for w in ["国保", "国民健康保険", "後期高齢者医療", "要介護認定", "保育所", "印鑑証明", "課税証明",
              "納税証明", "転居", "粗大ごみ", "水道"]:
        assert w in vocab
    assert "国保税" not in vocab and "後期高齢" not in vocab  # asr_exclude
    readings = {w.surface: w.reading for w in load_site_words("etajima", SERVER_ROOT / "context")
                if w.reading and w.surface in ("国保", "後期高齢", "住民票")}
    assert readings == {"国保": "コクホ", "後期高齢": "コウキコウレイ"}  # 読みが自明でないものだけ


async def test_refer_unspecified_department_is_saved_without_naming(tmp_path: Path) -> None:
    llm = StubLLM([["それは大変ですね。<propose:refer>"]], chat_reply=_refer_json(department="保健医療課"))
    c = _conv(llm, tmp_path)
    await _say(c, "ちょっと困ったことがあって")
    out = await _say(c, "お願い")
    assert out[1].startswith("担当の方に伝えておきました")  # 課名は言わない、聞き返さない
    (ref,) = ReferralStore(tmp_path, "etajima", 30, csv_export=True).list(datetime.now(JST).date())
    assert ref.department == "未特定"  # LLM が課を挙げても使わない
    assert c.phase == "listen"


async def test_refer_ambiguous_department_is_unspecified(tmp_path: Path) -> None:
    c = _conv(StubLLM([["そうですか。<propose:refer>"]], chat_reply=_refer_json()), tmp_path)
    await _say(c, "国保と住民票のことで聞きたいことがあって")
    await _say(c, "お願い")
    (ref,) = ReferralStore(tmp_path, "etajima", 30, csv_export=True).list(datetime.now(JST).date())
    assert ref.department == "未特定"


async def test_refer_summary_uses_only_current_topic(tmp_path: Path) -> None:
    class Spy(StubLLM):
        def __init__(self) -> None:
            super().__init__([["手元にありません。<propose:lookup>"], ["保健医療課です。<propose:refer>"]],
                             chat_reply=_refer_json())
            self.chat_inputs: list[str] = []

        def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> str:
            self.chat_inputs.append(messages[-1]["content"])
            return super().chat(model, messages, **kw)

    llm = Spy()
    c = _conv(llm, tmp_path, StubLookup(LookupResult(found=False)))
    await _say(c, "今週末は道の駅で何かやってますか")
    await _say(c, "いらない")
    await _say(c, "国保の切り替えで聞きたいことがあって")
    await _say(c, "お願いします")
    transcript = llm.chat_inputs[0]
    assert "国保の切り替え" in transcript and "お願いします" in transcript
    assert "道の駅" not in transcript and "いらない" not in transcript  # 前の話題を混ぜない


async def test_staff_can_assign_department(tmp_path: Path) -> None:
    from src.edge_server import StaffPanel, create_app

    store = ReferralStore(tmp_path, "etajima", 30, csv_export=False)
    ref = store.create(_fields(department="未特定"))
    panel = StaffPanel(store, VisitLog(tmp_path, "etajima"), StaffConfig(password="pw"),
                       departments=("市民課", "保健医療課"))
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none", panel), client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        auth = ("staff", "pw")
        assert (await c.get("/staff/api/departments", auth=auth)).json() == ["市民課", "保健医療課"]
        assert (await c.get("/staff/api/stats", auth=auth)).json()["unspecified"] == 1
        url = f"/staff/api/referrals/{ref.created_at[:10]}/{ref.id}"
        assert (await c.post(url, auth=auth, json={"department": "勝手な課"})).json()["department"] == "未特定"
        assert (await c.post(url, auth=auth, json={"department": "保健医療課"})).json()["department"] == "保健医療課"
        assert (await c.get("/staff/api/stats", auth=auth)).json()["unspecified"] == 0
