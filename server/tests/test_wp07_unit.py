"""WP07: 連絡方法と電話番号・保存の後の聞き取り再開・根拠の無い期限と金額・定型の一言の履歴・カメラのデバッグ表示。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.camera import PartySnapshot, PartyState
from src.config import DialogueConfig
from src.dialogue import Dialogue, grounding_overlap
from src.knowledge import Reference, SearchResult
from src.site_context import EmergencyLine, Facility
from tests.test_actions_unit import (
    PHRASED,
    FakeClock,
    R,
    StubLLM,
    _conv,
    _refer_json,
    _saved,
    _say,
    _to_ask_phone,
)
from tests.test_camera_unit import _loop
from tests.test_dialogue_unit import StubLLM as StreamLLM
from tests.test_dialogue_unit import _collect


def _c(tmp_path: Path, clock: FakeClock | None = None) -> Any:
    return _conv(StubLLM([["国保の切り替えですね。<propose:refer>"], ["ほかに何か。"]],
                         chat_reply=[PHRASED, _refer_json(), "{}"]), tmp_path, clock=clock)


# ------------------------------------------------------------------
# 1) 連絡方法と電話番号
# ------------------------------------------------------------------


@pytest.mark.parametrize(("answer", "expected"), [
    ("電話でお願いします", "電話"), ("携帯にかけてください", "電話"), ("折り返しもらえますか", "電話"),
    ("ここで待ちます", "窓口で待つ"), ("このまま待っています", "窓口で待つ"),
    ("後日また来ます", "後日来庁"), ("来週出直します", "後日来庁"),
])
def test_contact_rules(tmp_path: Path, answer: str, expected: str) -> None:
    assert _c(tmp_path)._contact_choices(answer) == [expected]


async def test_unclear_contact_is_confirmed_as_phone(tmp_path: Path) -> None:
    c = _c(tmp_path)
    await _to_ask_phone(c)
    # 「ちょっと待ってください」は窓口で待つではない。判定できないので電話か確かめる
    assert await _say(c, "えっと、ちょっと待ってください") == [R["confirm_contact"]]
    assert c.phase == "confirm_contact" and c.private_label == "ご連絡方法"
    assert await _say(c, "はい") == [R["ask_phone"]]  # 電話: 番号を伺う（保存しない）
    assert c.phase == "ask_phone"


async def test_ambiguous_contact_no_then_other_method(tmp_path: Path) -> None:
    c = _c(tmp_path)
    await _to_ask_phone(c)
    assert await _say(c, "電話か、また来るか") == [R["confirm_contact"]]  # 2 つに当たる
    out = await _say(c, "いいえ、ここで待ちます")
    assert c.last_receipt is not None and c.last_receipt in out[0]  # 窓口で待つ: 電話番号は聞かずに保存
    assert _saved(tmp_path).contact_pref == "窓口で待つ"


async def test_phone_contact_is_not_saved_without_number(tmp_path: Path) -> None:
    c = _c(tmp_path)
    await _to_ask_phone(c)
    assert await _say(c, "電話で") == [R["ask_phone"]]
    assert c.last_receipt is None  # まだ保存しない
    out = await _say(c, "番号は言いたくないです")
    assert out[0].startswith(R["phone_at_counter"])  # 番号なしで保存し、窓口で伺うと伝える
    ref = _saved(tmp_path)
    assert ref.contact_pref == "電話" and ref.phone is None


async def test_wrong_readback_asks_again_once(tmp_path: Path) -> None:
    c = _c(tmp_path)
    await _to_ask_phone(c)
    await _say(c, "電話で")
    assert await _say(c, "090-1234-5678") == [R["confirm_phone"].format(phone="0 9 0、1 2 3 4、5 6 7 8")]
    assert await _say(c, "違います") == [R["phone_reask"]]  # 電話なので 1 回だけ言い直してもらう
    assert c.last_receipt is None and c.phase == "ask_phone"
    assert await _say(c, "080-1111-2222") == [R["confirm_phone"].format(phone="0 8 0、1 1 1 1、2 2 2 2")]
    await _say(c, "はい")
    assert _saved(tmp_path).phone == "080-1111-2222"


# ------------------------------------------------------------------
# 2) 保存の後の聞き取り再開
# ------------------------------------------------------------------


async def test_reopen_phone_after_save_updates_same_referral(tmp_path: Path) -> None:
    clock = FakeClock()
    c = _c(tmp_path, clock)
    await _to_ask_phone(c)
    await _say(c, "ここで待ちます")  # 電話番号は聞かずに保存
    first = _saved(tmp_path)
    assert first.phone is None
    clock.t += 15
    out = await _say(c, "電話番号聞かなくていいの？")
    assert out == [R["reopen"] + R["ask_phone"]] and c.phase == "ask_phone"
    await _say(c, "090-1234-5678")
    out = await _say(c, "はい")
    assert out == [R["updated"].format(number=first.receipt)]
    ref = _saved(tmp_path)  # 同じ紹介状（1 件のまま）が書き直された
    assert ref.id == first.id and ref.phone == "090-1234-5678" and ref.contact_pref == "電話"


async def test_reopen_name_and_receipt_number_is_not_a_trigger(tmp_path: Path) -> None:
    clock = FakeClock()
    c = _c(tmp_path, clock)
    await _to_ask_phone(c)
    await _say(c, "ここで待ちます")
    clock.t += 15
    assert c._reopen_item("受付番号は何番でしたっけ") is None  # 受付番号は電話番号ではない
    assert await _say(c, "名前は言わなくていいんですか") == [R["reopen"] + R["ask_name"]]
    assert await _say(c, "山田です") == [R["confirm_name"].format(name="山田")]
    out = await _say(c, "はい")
    assert out == [R["updated"].format(number=c.last_receipt)]  # 地区へは進まない
    assert _saved(tmp_path).name == "山田"


# ------------------------------------------------------------------
# 3) 根拠の無い期限・金額・支払方法
# ------------------------------------------------------------------


class _KB:
    def __init__(self, refs: list[Reference]) -> None:
        self.refs = refs

    def search(self, query: str, uses: Any = None) -> SearchResult:
        return SearchResult(list(self.refs), 1.0)


async def test_ungrounded_deadline_and_payment_are_replaced_once() -> None:
    ref = Reference("国民健康保険税の納付について", "https://x", "国民健康保険税は年10回に分けて納めます。", 0.9, 3.0)
    d = Dialogue(StreamLLM(["国保税のお支払いですね。", "今月の月末までに納めてください。", "口座振替や納付書でお支払いできます。",
                            "保健医療課でご案内します。"]), DialogueConfig(), knowledge=_KB([ref]))
    out = await _collect(d, "国保税の払い方を教えて")
    assert out == ["国保税のお支払いですね。", DialogueConfig().grounding_caution, "保健医療課でご案内します。"]
    assert [x["reason"] for x in d.last_dropped] == ["根拠の無い期限・金額・支払方法"] * 2


async def test_grounded_amount_and_facility_deadline_are_kept() -> None:
    ref = Reference("住民票の写し", "https://x", "住民票の写しの手数料は1通300円です。", 0.9, 3.0)
    fac = Facility(name="江田島市役所", location_phrase="1階ロビー",
                   notes=("延長窓口は予約制です。前日までに市民課へ電話してください",),
                   emergency_lines=(EmergencyLine("フリーダイヤル", "0120-95-0178"),))
    d = Dialogue(StreamLLM(["住民票の写しの手数料は1通300円です。", "延長窓口は前日までに市民課へ電話してください。"]),
                 DialogueConfig(), knowledge=_KB([ref]), facility=fac)
    assert await _collect(d, "住民票の手数料は") == ["住民票の写しの手数料は1通300円です。",
                                                  "延長窓口は前日までに市民課へ電話してください。"]
    assert grounding_overlap("今月の月末までに", "国民健康保険税は年10回に分けて納めます。") < 0.3


# ------------------------------------------------------------------
# 4) 沈黙・別れの一言を履歴に
# ------------------------------------------------------------------


async def test_silence_and_farewell_lines_go_into_history(tmp_path: Path) -> None:
    c = _c(tmp_path)
    c.note_assistant_line("ゆっくりで大丈夫ですよ。")  # 履歴が空: 1 ターンにする
    assert c.dialogue.history[-1][1] == "ゆっくりで大丈夫ですよ。"
    await _say(c, "今日は寒いですね")
    c.note_assistant_line("お気をつけて。またどうぞ。")  # 直前の応答の後ろに足す
    assert c.dialogue.history[-1][1].endswith("お気をつけて。またどうぞ。")
    assert len(c.dialogue.history) == 2


# ------------------------------------------------------------------
# 1) の補足: 聞き取りの途中は、顔が下を向いていても聞く
# ------------------------------------------------------------------


async def test_listen_while_present_during_intake(tmp_path: Path) -> None:
    from src.actions import Pending

    app, _edge = _loop()
    assert app._camera is not None
    app._camera.push_jpeg(b"\xff\xd8")
    app._camera.tracker.snapshot = PartySnapshot(PartyState.PRESENT, 0, False)
    assert not app._camera_listening()  # ふだんは輪が無ければ聞かない
    app._conversation._pending = Pending("refer", "国保", "ja", phase="ask_phone")
    assert app._camera_listening()  # 聞き取りの途中は PRESENT でも聞く（携帯を見ながら番号を言う）
    app._camera.tracker.snapshot = PartySnapshot(PartyState.NONE, 0, False)
    assert not app._camera_listening()
    await app._tts.close()


# ------------------------------------------------------------------
# 6) カメラのデバッグ表示
# ------------------------------------------------------------------


async def test_debug_page_loads_frames_sequentially(tmp_path: Path) -> None:
    from src.edge_server import create_app

    frame: list[bytes | None] = [None]
    app = create_app(tmp_path, debug_frame=lambda: frame[0], debug_state=lambda: {"state": "NONE"})
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        page = (await client.get("/edge/debug")).text
        assert "createObjectURL" in page and 'cache: "no-store"' in page and "setTimeout(loadFrame" in page
        assert (await client.get("/edge/debug.jpg")).status_code == 404  # 初回はまだ無い
        frame[0] = b"\xff\xd8jpeg"
        r = await client.get("/edge/debug.jpg")
        assert r.status_code == 200 and r.content == b"\xff\xd8jpeg" and r.headers["cache-control"] == "no-store"


async def test_umbrella_follows_the_day_and_tag_leak_is_dropped() -> None:
    from datetime import datetime

    from src.knowledge import JST

    class _W:
        def lines(self, now: datetime) -> list[str]:
            return ["今日（10月22日）の広島県南部は「晴れ」の予報です。降水確率は、18時から24時が0パーセントです。",
                    "明日（10月23日）の広島県南部は「晴れ　後　雨」の予報です。降水確率は最大50パーセントです。"]

    def noon() -> datetime:
        return datetime(2026, 10, 22, 12, 0, tzinfo=JST)

    d = Dialogue(StreamLLM(["今日は晴れの予報です。", "お出かけの際は傘を持ち歩くとよいかもしれませんね。"]),
                 DialogueConfig(), weather=_W(), now=noon)  # type: ignore[arg-type]
    assert await _collect(d, "今日の天気は") == ["今日は晴れの予報です。"]  # 今日は雨が無いので傘をすすめない
    d = Dialogue(StreamLLM(["明日は夜遅く雨の予報です。", "傘をお持ちください。"]), DialogueConfig(), weather=_W(),
                 now=noon)  # type: ignore[arg-type]
    assert await _collect(d, "明日は雨が降りますか") == ["明日は夜遅く雨の予報です。", "傘をお持ちください。"]
    d = Dialogue(StreamLLM(["オレンジドクターに相談してみてください。", "提案する：参照する"]), DialogueConfig())
    assert await _collect(d, "物忘れが") == ["オレンジドクターに相談してみてください。"]


async def test_comma_fragment_about_deadline_is_judged_with_its_rest() -> None:
    ref = Reference("国民健康保険税の納付について", "https://x", "国民健康保険税は年10回に分けて納めます。", 0.9, 3.0)
    d = Dialogue(StreamLLM(["国民健康保険税の納期限は、", "毎年1月31日までです。", "この期限までに必ず納めてくださいね。",
                            "税務課でご案内します。"]),
                 DialogueConfig(), knowledge=_KB([ref]))
    out = await _collect(d, "国保税はいつまでに払えば")
    assert out == [DialogueConfig().grounding_caution, "税務課でご案内します。"]  # 頭の断片も、それを指す文も残らない
    ok = Reference("住民票などの郵便請求", "https://x", "| 種類 | 手数料 |\n| 住民票の写し | 200円 |", 0.9, 3.0)
    d = Dialogue(StreamLLM(["住民票の写しの手数料は、", "1通200円です。"]), DialogueConfig(), knowledge=_KB([ok]))
    assert await _collect(d, "住民票はいくら") == ["住民票の写しの手数料は、1通200円です。"]  # 資料にある金額は言う


async def test_medical_judgment_with_naried_form_is_dropped() -> None:
    d = Dialogue(StreamLLM(["それは心配ですね。", "認知症の初期症状の一つにもなり得ますが、", "まずは医師に相談してみてください。"]),
                 DialogueConfig())
    out = await _collect(d, "最近、物忘れがひどくて心配なんです")
    assert "".join(out) == "それは心配ですね。まずは医師に相談してみてください。"
