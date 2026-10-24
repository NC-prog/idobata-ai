"""用件の聞き分け（WP13 E）: 複数の手続きが同程度に当たったら 1 問だけ聞き、返事で 1 件に決める。"""

from __future__ import annotations

import json
from pathlib import Path

from src.actions import Conversation
from src.clarify import Clarifier, load_clarify_rules
from src.config import SERVER_ROOT, ActionsConfig, DialogueConfig, ProcedureConfig, ReferralConfig
from src.dialogue import Dialogue, speech_text
from src.procedure_kb import ProcedureBook, ProcedureRecord
from src.procedures import Procedures, load_procedures
from src.referrals import ReferralStore, VisitLog
from tests.procedure_helpers import seeded_procedures
from tests.test_actions_unit import FAC, TAGS, FakeClock, StubLLM, _noon

RULES = load_clarify_rules(SERVER_ROOT / "context" / "actions.yaml")


def _procs(tmp_path: Path) -> Procedures:
    d = tmp_path / "procs"
    seeded_procedures(d)  # 国保の加入・マイナンバーカードの紛失・介護保険（承認済み）
    book = ProcedureBook(d)
    book.save(ProcedureRecord(id="kokuho_withdraw", name="国民健康保険の脱退・資格確認書", aliases=("国民健康保険", "国保"),
                              when=("やめ", "脱退", "抜け", "就職", "資格確認書", "保険証", "再発行"),
                              department="保健医療課", status="approved", approved_by="test", approved_at="2026-10-24"))
    book.save(ProcedureRecord(id="jido_teate", name="児童手当", aliases=("児童手当",), department="子育て支援課",
                              status="approved", approved_by="test", approved_at="2026-10-24"))
    book.save(ProcedureRecord(id="kouki_korei", name="後期高齢者医療", aliases=("後期高齢", "後期高齢者"), status="draft"))
    return load_procedures(SERVER_ROOT / "context", d)


def _conv(tmp_path: Path, llm: StubLLM) -> Conversation:
    procs = _procs(tmp_path)
    d = Dialogue(llm, DialogueConfig(), facility=FAC, enabled_actions=("lookup", "refer"), now=_noon, procedures=procs)
    return Conversation(d, llm, "/models/llm", ActionsConfig(enabled=("lookup", "refer")), ReferralConfig(), FAC, TAGS,
                        referrals=ReferralStore(tmp_path, "etajima", 30, csv_export=True),
                        visits=VisitLog(tmp_path, "etajima"), procedures=procs, now=_noon, clock=FakeClock())


async def _say(c: Conversation, text: str) -> list[str]:
    return [s async for s in c.respond(text)]


def test_find_asks_only_when_tied(tmp_path: Path) -> None:
    c = Clarifier(_procs(tmp_path), RULES, ProcedureConfig())
    q = c.find("国保のことで")
    assert q is not None and q.ids == ["kokuho_switch", "kokuho_withdraw", "term:国民健康保険税"]
    assert c.question(q).startswith("国保のことですね。国保に入る手続きか、国保をやめる手続きや保険証か、国保税か、どちらでしょう？")
    q = c.find("手当のことで")
    assert q is not None and q.ids == ["jido_teate", "term:児童扶養手当", "term:特別児童扶養手当"]
    q = c.find("保険証")
    assert q is not None and set(q.ids) == {"kokuho_withdraw", "kouki_korei"}
    # 1 件に絞れる発話では聞かない
    for text in ("国保に入りたい", "国保税のことで", "国保の保険証をなくした", "児童手当について", "マイナンバーカードをなくした",
                 "転入の手続き", "住民票はどこで取れますか", "子どもが生まれたので児童手当の申請をしたいのですが",
                 "代わりに申請しといてよ", "子どもの学校かわるんじゃけど何すりゃええ？", "ゴミの日っていつなん？"):
        assert c.find(text) is None, text


def test_question_numbers_are_caption_only(tmp_path: Path) -> None:
    c = Clarifier(_procs(tmp_path), RULES, ProcedureConfig())
    q = c.find("国保のことで")
    assert q is not None
    line = c.question(q)
    assert "（1. 国保に入る手続き　2. 国保をやめる手続きや保険証　3. 国保税）" in line
    assert "1." not in speech_text(line)  # 読み上げは番号を言わない


def test_pick_reply_by_words_order_and_other(tmp_path: Path) -> None:
    c = Clarifier(_procs(tmp_path), RULES, ProcedureConfig())
    q = c.find("国保のことで")
    assert q is not None
    def pick(t: str) -> str | None:
        r = c.pick(q, t)
        return r if isinstance(r, str) or r is None else r.id

    assert pick("入りたいんです") == "kokuho_switch"
    assert pick("やめるほうです") == "kokuho_withdraw"
    assert pick("保険証のことで") == "kokuho_withdraw"
    assert pick("税金のほう") == "term:国民健康保険税"
    assert pick("二つ目") == "kokuho_withdraw"
    assert pick("最後の") == "term:国民健康保険税"
    assert pick("ほかのことです") == "other"
    assert pick("うーん") is None
    assert pick("国保です") is None  # 鍵そのものでは決まらない


async def test_clarify_then_answer_with_the_one(tmp_path: Path) -> None:
    llm = StubLLM([["保健医療課で手続きできます。"]])
    c = _conv(tmp_path, llm)
    first = await _say(c, "国保のことで")
    assert first[0].startswith("国保のことですね。")
    assert c.expects_short_reply  # 返事待ち（VAD を下げ、言語を固定する）
    assert llm.stream_calls == []  # 問いは LLM を通さない
    second = await _say(c, "やめるほうです")
    assert "保健医療課" in "".join(second)
    assert [r.id for r in c.dialogue.last_procedures] == ["kokuho_withdraw"]
    assert all(r.trust == "procedure" for r in c.dialogue.last_references)  # 決まった 1 件だけ（パックは渡さない）
    assert c.turn_info["clarify"] == "聞き分けた（候補 3 件 → kokuho_withdraw）"
    # 短い返事は、LLM には決まった手続きの名前で渡す（「二つ目ですね」とおうむ返ししない）
    assert llm.stream_calls[-1][-1]["content"].endswith("国保をやめる手続きや保険証のことです。")
    # 同じ用件ではもう聞かない
    third = await _say(c, "国保のことで")
    assert not third[0].startswith("国保のことですね。")
    await c.end_session()
    visit = json.loads(next((tmp_path / "visits").rglob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert {"action": "clarify", "outcome": "kokuho_withdraw", "candidates": "3"} in visit["proposals"]


async def test_no_clarify_when_one_fits(tmp_path: Path) -> None:
    llm = StubLLM([["保健医療課で手続きできます。"]])
    c = _conv(tmp_path, llm)
    said = await _say(c, "国保に入りたい")
    assert not said[0].startswith("国保のことですね。") and llm.stream_calls


async def test_clarify_reask_then_unspecified_referral(tmp_path: Path) -> None:
    llm = StubLLM([])
    c = _conv(tmp_path, llm)
    await _say(c, "手当のことで")
    again = await _say(c, "うーん")
    assert again[0].startswith("すみません、児童手当か児童扶養手当か特別児童扶養手当か、どちらでしょう？")
    last = await _say(c, "えっと")
    assert last == ["手続きの詳しいことは、担当の課でご案内します。このこと、担当の課に伝えておきましょうか？"]
    assert c.turn_info["clarify"] == "聞き分けた（候補 3 件 → 未決）"
    await _say(c, "お願いします")
    assert c._pending is not None and c._pending.department == "未特定"


async def test_clarify_by_order_and_other(tmp_path: Path) -> None:
    c = _conv(tmp_path, StubLLM([["子育て支援課でご案内します。"]]))
    await _say(c, "手当のことで")
    await _say(c, "一つ目です")
    assert [r.id for r in c.dialogue.last_procedures] == ["jido_teate"]
    c2 = _conv(tmp_path / "b", StubLLM([]))
    await _say(c2, "保険証")
    assert await _say(c2, "ほかのことです") == ["わかりました。どんなことでしょう？"]


async def test_no_clarify_when_hurried(tmp_path: Path) -> None:
    llm = StubLLM([["保健医療課です。"]])
    c = _conv(tmp_path, llm)
    said = await _say(c, "急いでるんだけど国保のことで")
    assert not said[0].startswith("国保のことですね。")


async def test_clarified_intake_is_one_shorter(tmp_path: Path) -> None:
    # 聞き分けの 1 問と聞き取りは合わせて 3 問（国保の加入の聞き取り 3 問 → 2 問）
    llm = StubLLM([["保健医療課で手続きできます。<propose:refer>"]], chat_reply="{}")
    c = _conv(tmp_path, llm)
    await _say(c, "国保のことで")
    said = await _say(c, "入りたいんです")
    assert said[-1] == "このこと、担当の課に伝えておきましょうか？"
    await _say(c, "お願いします")
    p = c._pending
    assert p is not None and p.department == "保健医療課" and len(p.intake) == 2


async def test_repeated_question_is_answered_not_reasked(tmp_path: Path) -> None:
    # 聞き分けの問いに答えず同じ問いを言い直した（意地悪 20 番の形。場所の問いは聞き分けないので、中身の問いで見る）: 聞き直しを重ねず、ふつうに答える
    llm = StubLLM([["国保の手続きは保健医療課（2階）で承ります。"]] * 4)
    c = _conv(tmp_path, llm)
    first = await _say(c, "国保の手続きって何が必要ですか")
    assert first[0].startswith("国保のことですね。")
    second = await _say(c, "国保の手続きって何が必要ですか")
    assert "保健医療課" in "".join(second)
    assert c.turn_info["clarify"] == "聞き分けた（候補 3 件 → 未決（別の話））"
    for _ in range(3):
        said = await _say(c, "国保の手続きって何が必要ですか")
        assert not any(x.startswith(("国保のことですね", "すみません、国保に入る")) for x in said)  # 聞き分けを重ねない


async def test_question_while_proposal_pending_is_a_new_topic(tmp_path: Path) -> None:
    # 紹介状の提案の返事待ちに、問いの形（12 字以下でも）で言われたら、返事ではなく別の話として答える
    llm = StubLLM([["保健医療課です。<propose:refer>"], ["住民票は市民課（1階）で取れます。"]])
    c = _conv(tmp_path, llm)
    await _say(c, "国保に入りたいんですが")
    assert c.pending_action == "refer"
    said = await _say(c, "住民票はどこですか")
    assert "市民課" in "".join(said) and c.pending_action != "refer"


async def test_place_question_with_one_department_is_answered(tmp_path: Path) -> None:
    # 場所を聞く問いで、候補（国保の加入・脱退）の課が同じ保健医療課なら聞き分けずに答える。中身を聞いたら聞き分ける
    llm = StubLLM([["保健医療課（2階）で承ります。"]])
    c = _conv(tmp_path, llm)
    said = await _say(c, "国保の手続きはどこですか")
    assert "保健医療課" in "".join(said) and not said[0].startswith("国保のことですね。")
    c2 = _conv(tmp_path / "b", StubLLM([]))
    said = await _say(c2, "国保の手続きって何が必要ですか")
    assert said[0].startswith("国保のことですね。")
    # 場所の問いでも、課が分かれる候補（児童手当＝子育て支援課、特別児童扶養手当＝社会福祉課）なら聞き分ける
    c3 = _conv(tmp_path / "c", StubLLM([]))
    said = await _say(c3, "手当の窓口はどこですか")
    assert said[0].startswith("手当のことですね。")
