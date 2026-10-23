"""WP09: 手続きナレッジ（職員が承認する正解表）の読み込み・担当課・参考資料への注入・再確認の検知。"""

from __future__ import annotations

import os
import time
from pathlib import Path

from src.config import SERVER_ROOT, DialogueConfig
from src.dialogue import Dialogue
from src.persona import build_turn_message
from src.procedure_kb import (
    PROCEDURE_TRUST,
    ProcedureBook,
    check_sources,
    content_hash,
    dump_record,
    load_seeds,
    record_from_dict,
)
from src.procedures import load_procedures
from src.site_context import load_facility
from tests.procedure_helpers import SEEDS, seeded_procedures
from tests.test_dialogue_unit import _NOON, StubLLM, _collect

FAC = load_facility("etajima", SERVER_ROOT / "context")


def _rec(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "juminhyo", "name": "住民票の写しの交付", "aliases": ["住民票"], "department": "市民課",
        "bring": ["本人確認書類"], "steps": ["市民課の窓口で申請書を書く", "本人確認書類を見せる"],
        "deadline": None, "fee": "1通300円",
        "faq": [{"q": "住民票はいくら", "a": "1通300円です。"}],
        "sources": [{"url": "https://www.city.etajima.hiroshima.jp/cms/articles/show/9557", "checked_at": "2026-10-23"}],
        "status": "draft",
    }
    base.update(kw)
    return base


def _book(tmp_path: Path, *recs: dict[str, object]) -> Path:
    d = tmp_path / "procs"
    book = ProcedureBook(d)
    for r in load_seeds(SEEDS, "etajima"):
        book.save(r)
    for r in recs:
        book.save(record_from_dict(r))
    return d


# ---- 形式 -----------------------------------------------------------------


def test_record_roundtrip_and_seeds_are_approved() -> None:
    r = record_from_dict(_rec())
    assert record_from_dict(__import__("yaml").safe_load(dump_record(r))) == r
    seeds = {s.id: s for s in load_seeds(SEEDS, "etajima")}
    assert set(seeds) == {"kokuho_switch", "mynumber_lost", "kaigo_nintei"}
    assert all(s.status == "approved" and s.approved_at for s in seeds.values())
    assert seeds["kokuho_switch"].department == "保健医療課"
    assert {s.id: s.department for s in load_seeds(SEEDS, "akiota")}["kaigo_nintei"] == "健康福祉課"


def test_bad_file_is_skipped(tmp_path: Path) -> None:
    d = _book(tmp_path)
    (d / "broken.yaml").write_text("id: Bad Id\nname: x\n", encoding="utf-8")
    book = ProcedureBook(d)
    assert len(book.records()) == 3 and "broken.yaml" in book.errors


# ---- 突き合わせと担当課 ------------------------------------------------------


def test_match_containment_and_when(tmp_path: Path) -> None:
    d = _book(tmp_path, _rec(id="mynumber_apply", name="マイナンバーカードの申請", aliases=["マイナンバーカード"],
                             department="市民課", fee=None, faq=[], sources=[]))
    procs = load_procedures(SERVER_ROOT / "context", d)
    assert [r.id for r in procs.matched("国保の切り替えをしたい")] == ["kokuho_switch"]
    assert procs.matched("国保税を払いたい") == []  # 「国保」は「国保税」の中
    # 同じ語で当たったら、when で絞った手続き（紛失）を選ぶ
    assert [r.id for r in procs.matched("マイナンバーカードをなくしてしまって")] == ["mynumber_lost"]
    assert [r.id for r in procs.matched("マイナンバーカードを作りたい")] == ["mynumber_apply"]


def test_departments_from_approved_knowledge_and_alias_table(tmp_path: Path) -> None:
    procs = seeded_procedures(tmp_path / "p")
    assert procs.departments_for("国保に入りたい", FAC) == ["保健医療課"]
    assert procs.departments_for("国保税を払いたい", FAC) == ["税務課"]  # 別名表と handles の規則のまま
    assert procs.departments_for("国保と住民票のこと", FAC) == ["市民課", "保健医療課"]  # 一意でない → 未特定
    assert procs.departments_for("介護の申請を", FAC) == ["高齢介護課"]


def test_draft_department_is_not_used(tmp_path: Path) -> None:
    # 下書きの課（作り話かもしれない）では決めない。別名表の規則に倒す
    d = _book(tmp_path, _rec(id="passport", name="パスポートの申請", aliases=["パスポート"], department="総務課"))
    procs = load_procedures(SERVER_ROOT / "context", d)
    assert procs.departments_for("パスポートを作りたい", FAC) == ["市民課"]
    approved = load_procedures(SERVER_ROOT / "context", _book(tmp_path / "b", _rec(
        id="passport", name="パスポートの申請", aliases=["パスポート"], department="総務課", status="approved",
        approved_at="2026-10-23T10:00:00+09:00")))
    assert approved.departments_for("パスポートを作りたい", FAC) == ["総務課"]


def test_intake_comes_from_approved_knowledge_only(tmp_path: Path) -> None:
    d = _book(tmp_path, _rec(intake=[{"key": "n", "question": "何通要るか"}]))
    procs = load_procedures(SERVER_ROOT / "context", d)
    s = procs.intake_for("国保の切り替えで", FAC)
    assert s is not None and s.questions[0].startswith("切り替えのきっかけ")
    assert s.bring_for("会社を辞めた") == "前の健康保険の資格喪失証明書"
    assert s.bring_for("転入してきた") == "前の市町村の転出証明書"
    assert procs.intake_for("住民票がほしい", FAC) is None  # 下書きの聞き取りは使わない
    assert procs.tags_for("国保の切り替え", FAC) == ["保険・年金"]


def test_reload_when_file_changes(tmp_path: Path) -> None:
    d = _book(tmp_path, _rec())
    procs = load_procedures(SERVER_ROOT / "context", d)
    assert procs.usable_matched("住民票がほしい") == []
    path = d / "juminhyo.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("status: draft", "status: approved")
                    .replace("approved_at: null", "approved_at: '2026-10-23T10:00:00+09:00'"), encoding="utf-8")
    t = time.time() + 5
    os.utime(path, (t, t))
    assert [r.id for r in procs.usable_matched("住民票がほしい")] == ["juminhyo"]


# ---- 対話への注入 ------------------------------------------------------------


def test_turn_message_puts_procedures_first() -> None:
    from src.knowledge import Reference

    r = record_from_dict(_rec()).to_reference("市民課（1階）")
    page = Reference("住民票などの窓口発行", "https://x", "本文", 0.9, 1.0)
    msg = build_turn_message(DialogueConfig(), FAC, [r, page], "住民票", "【市の案内（承認済み）】\n")
    assert msg.index("【市の案内（承認済み）】") < msg.index("【参考資料】")
    assert "[1] 手続き: 住民票の写しの交付" in msg and "[2] (出典: 住民票などの窓口発行)" in msg
    assert r.trust == PROCEDURE_TRUST


async def test_draft_is_not_injected_and_falls_back_to_department(tmp_path: Path) -> None:
    procs = load_procedures(SERVER_ROOT / "context", _book(tmp_path, _rec()))
    llm = StubLLM(["住民票の写しは本人確認書類で取れますよ。"])
    d = Dialogue(llm, DialogueConfig(), facility=FAC, procedures=procs, now=_NOON)
    out = await _collect(d, "住民票はどうやって取るんですか")
    assert "【市の案内（承認済み）】" not in llm.calls[0][1][-1]["content"]
    assert "1通300円" not in llm.calls[0][1][-1]["content"]
    assert out[-1] == "手続きの詳しいことは、市民課でご案内します。"
    assert [r.id for r in d.last_procedures] == ["juminhyo"]
    # 同じ会話で同じ手続きには繰り返さない
    llm.deltas = ["そうですね。"]
    assert await _collect(d, "住民票って何が要りますか") == ["そうですね。"]


async def test_draft_fallback_not_added_when_reply_already_points_to_department(tmp_path: Path) -> None:
    procs = load_procedures(SERVER_ROOT / "context", _book(tmp_path, _rec()))
    d = Dialogue(StubLLM(["持ち物は市民課でご案内しますので、お声がけください。"]), DialogueConfig(),
                 facility=FAC, procedures=procs, now=_NOON)
    assert "".join(await _collect(d, "住民票は何が要りますか")) == "持ち物は市民課でご案内しますので、お声がけください。"


async def test_approved_is_injected_attributed_and_grounds_fee(tmp_path: Path) -> None:
    procs = load_procedures(SERVER_ROOT / "context", _book(tmp_path, _rec(
        status="approved", approved_by="staff", approved_at="2026-10-23T10:00:00+09:00")))
    llm = StubLLM(["住民票の写しは市民課（1階）で取れます。", "手数料は1通300円です。[出典:1]",
                   "本人確認書類をお持ちください。"])
    d = Dialogue(llm, DialogueConfig(), facility=FAC, procedures=procs, now=_NOON)
    out = await _collect(d, "住民票はいくらですか")
    content = llm.calls[0][1][-1]["content"]
    assert "【市の案内（承認済み）】" in content and "手数料・金額: 1通300円" in content
    assert "担当: 市民課（1階）" in content
    # 期限・金額の安全装置: 承認済みの fee は「資料にある」扱い。「市の案内では」は一度だけ
    assert out == ["住民票の写しは市民課（1階）で取れます。", "市の案内では、手数料は1通300円です。",
                   "本人確認書類をお持ちください。"]
    assert [r.chunk_id for r in d.last_cited] == ["procedure:juminhyo"]
    assert not any("ホームページ" in x for x in out)  # 手続きナレッジはホームページの一言に数えない


async def test_ungrounded_fee_still_dropped_for_draft(tmp_path: Path) -> None:
    procs = load_procedures(SERVER_ROOT / "context", _book(tmp_path, _rec()))
    d = Dialogue(StubLLM(["住民票は1通300円です。"]), DialogueConfig(), facility=FAC, procedures=procs, now=_NOON)
    out = await _collect(d, "住民票はいくらですか")
    assert "300円" not in "".join(out)
    assert out[0] == DialogueConfig().grounding_caution


# ---- 再確認の検知 ------------------------------------------------------------


def test_check_sources_marks_recheck(tmp_path: Path) -> None:
    url = "https://www.city.etajima.hiroshima.jp/cms/articles/show/9557"
    h = content_hash(["本文A"])
    d = _book(tmp_path, _rec(status="approved", approved_at="2026-10-23T10:00:00+09:00", content_hash=h),
              _rec(id="draft_one", name="下書きの手続き", aliases=["下書き"], content_hash=h))
    book = ProcedureBook(d)
    assert check_sources(book, {url: ["本文A"]}) == []
    assert check_sources(book, {url: ["本文B"]}) == ["juminhyo"]  # 下書きは印を付けない
    r = book.get("juminhyo")
    assert r is not None and r.status == "recheck" and r.recheck_reason == "出典のページの本文が変わりました"
    assert r.content_hash == content_hash(["本文B"])
    assert book.usable(r)  # 既定は使い続ける（要再確認の印だけ）
    assert not ProcedureBook(d, keep_recheck=False).usable(r)
    assert book.clear_recheck("juminhyo").status == "approved"
    assert check_sources(book, {}) == ["juminhyo"]  # パックから消えた


# ---- 職員パネル「手続き」タブ ------------------------------------------------


async def test_staff_panel_edit_approve_and_dialogue_uses_it(tmp_path: Path) -> None:
    import httpx

    from src.config import StaffConfig
    from src.edge_server import StaffPanel, create_app
    from src.procedure_panel import ProcedurePanel
    from src.referrals import ReferralStore, VisitLog

    d = _book(tmp_path, _rec())
    procs = load_procedures(SERVER_ROOT / "context", d)
    assert procs.book is not None
    asked: list[str] = []

    async def runner(record: object, q: str) -> str:
        asked.append(q)
        return "市の案内では、1通300円です。"

    panel = StaffPanel(ReferralStore(tmp_path, "etajima", 30, csv_export=False), VisitLog(tmp_path, "etajima"),
                       StaffConfig(password="pw"), departments=tuple(x.name for x in FAC.departments),
                       procedures=ProcedurePanel(procs.book, tmp_path, tuple(x.name for x in FAC.departments), runner))
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none", panel), client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        auth = ("staff", "pw")
        assert (await c.get("/staff/api/procedures", auth=("staff", "x"))).status_code == 401
        rows = (await c.get("/staff/api/procedures?status=draft", auth=auth)).json()
        assert [r["id"] for r in rows] == ["juminhyo"] and rows[0]["checked_at"] == "2026-10-23"
        assert "手続きナレッジ" in (await c.get("/staff/procedures/", auth=auth)).text
        # 編集（持ち物の「品名 | 語」の書き方）
        r = await c.put("/staff/api/procedures/juminhyo", auth=auth, json={"fields": {
            "bring": ["本人確認書類", "委任状 | 代理, 代わり"], "staff_note": "窓口で番号札を取ってもらう"}})
        assert r.status_code == 200 and r.json()["bring_lines"] == ["本人確認書類", "委任状 | 代理、代わり"]
        bad = await c.put("/staff/api/procedures/juminhyo", auth=auth, json={"fields": {"status": "approved"}})
        assert bad.status_code == 422  # 状態は承認の操作でしか変えない
        assert (await c.put("/staff/api/procedures/juminhyo", auth=auth,
                            json={"fields": {"department": "存在しない課"}})).status_code == 422
        # 下書きのうちは対話に入らない
        llm = StubLLM(["住民票は市民課で取れます。"])
        dlg = Dialogue(llm, DialogueConfig(), facility=FAC, procedures=procs, now=_NOON)
        await _collect(dlg, "住民票はいくらですか")
        assert "【市の案内（承認済み）】" not in llm.calls[-1][1][-1]["content"]
        # 承認 → 承認者はログイン名。読み直しなしで次のターンから入る
        r = await c.post("/staff/api/procedures/juminhyo/approve", auth=auth, json={})
        assert r.json()["status"] == "approved" and r.json()["approved_by"] == "staff"
        dlg.reset()
        await _collect(dlg, "住民票はいくらですか")
        assert "職員から: 窓口で番号札を取ってもらう" in llm.calls[-1][1][-1]["content"]
        # 想定問答を試す・×は素材に
        tried = (await c.post("/staff/api/procedures/juminhyo/faq/try", auth=auth)).json()
        assert tried == [{"q": "住民票はいくら", "expected": "1通300円です。", "answer": "市の案内では、1通300円です。"}]
        r = await c.post("/staff/api/procedures/juminhyo/faq/review", auth=auth,
                         json={"q": "住民票はいくら", "answer": "x", "ok": False, "note": "金額が古い"})
        assert r.json() == {"written": True}
        assert "金額が古い" in next((tmp_path / "review").glob("procedures_*.jsonl")).read_text(encoding="utf-8")
        # 差し戻し
        r = await c.post("/staff/api/procedures/juminhyo/send_back", auth=auth, json={})
        assert r.json()["status"] == "draft" and r.json()["approved_at"] is None


# ---- 下書きの対象一覧と下書きの点検（scripts/draft_procedures.py） -----------------


def _draft_module() -> object:
    import importlib
    import sys

    sys.path.insert(0, str(SERVER_ROOT.parent / "scripts"))
    return importlib.import_module("draft_procedures")


def test_catalog_has_50_with_valid_tags_and_rule_departments() -> None:
    import yaml

    dp: object = _draft_module()
    tags = set(yaml.safe_load((SERVER_ROOT / "context" / "tags.yaml").read_text(encoding="utf-8"))["tags"])
    for site in ("etajima", "akiota"):
        fac = load_facility(site, SERVER_ROOT / "context")
        cat = dp.load_catalog(site, fac)  # type: ignore[attr-defined]
        assert len(cat) == 50 and len({c["id"] for c in cat}) == 50
        assert all(len(c["asks"]) == 2 and "〔市町〕" not in str(c) for c in cat)
        assert all(t in tags for c in cat for t in c.get("tags") or [])
        names = {d.name for d in fac.departments}
        for c in cat:
            dept, _ = dp.catalog_department(c, fac)  # type: ignore[attr-defined]
            assert dept is None or dept in names
    aki = load_facility("akiota", SERVER_ROOT / "context")
    by = {c["id"]: c for c in dp.load_catalog("akiota", aki)}  # type: ignore[attr-defined]
    # 支所と本庁の両方が handles を持つときは、この建物（本庁）の課
    assert dp.catalog_department(by["juminhyo"], aki)[0] == "税務住民課"  # type: ignore[attr-defined]
    assert dp.catalog_department(by["houkatsu"], aki)[0] == "健康福祉課"  # type: ignore[attr-defined]


def test_draft_grounding_rejects_fee_from_another_row() -> None:
    dp: object = _draft_module()
    src = ["| 住民票の写し | 200円 |\n| 戸籍謄本 | 450円 |\n本人確認書類が必要です"]
    words = ["戸籍謄本", "戸籍"]
    grounded = dp.grounded  # type: ignore[attr-defined]
    assert grounded("450円", src, "fee", words)
    assert not grounded("200円", src, "fee", words)  # 表の別の行（住民票）の金額
    assert not grounded("1通500円", src, "fee", words)  # 資料に無い数
    assert grounded("本人確認書類", src, "bring", words)
    assert not grounded("印鑑", src, "bring", words)


async def test_draft_fallback_not_doubled_when_reply_names_department(tmp_path: Path) -> None:
    procs = load_procedures(SERVER_ROOT / "context", _book(tmp_path, _rec()))
    d = Dialogue(StubLLM(["本人確認書類をお持ちください。", "何かあれば、市民課で詳しくご説明しますので、お声がけください。"]),
                 DialogueConfig(), facility=FAC, procedures=procs, now=_NOON)
    out = await _collect(d, "住民票は何が要りますか")
    assert not any("手続きの詳しいことは" in x for x in out)


def test_conditional_bring_is_written_with_condition() -> None:
    (r,) = [s for s in load_seeds(SEEDS, "etajima") if s.id == "kokuho_switch"]
    assert "前の健康保険の資格喪失証明書（退職・辞め・扶養・会社 の場合）" in r.facts_text()


# ---- 定型への切り替え（C） ------------------------------------------------------


async def test_crisis_reply_is_fixed_and_names_consult_places() -> None:
    llm = StubLLM(["それは大変ですね。"])
    d = Dialogue(llm, DialogueConfig(), facility=FAC, now=_NOON)
    out = await _collect(d, "もう死にたい")
    assert out[0] == "話してくださって、ありがとうございます。" and "110番" in out[1]
    assert out[2] == "社会福祉課（2階）や高齢介護課（2階）でも、相談を受けています。"
    assert "近くの職員を呼びます" in out[3] and d.last_safety == "crisis" and not llm.calls  # LLM を通さない


async def test_crisis_without_consult_place_falls_back_to_staff() -> None:
    from src.site_context import Facility as F

    d = Dialogue(StubLLM(), DialogueConfig(), facility=F(name="テスト役場", location_phrase="ロビー"), now=_NOON)
    out = await _collect(d, "夫に殴られるんです")
    assert len(out) == 3 and "近くの職員を呼びます" in out[-1]


async def test_crisis_interrupts_referral_intake(tmp_path: Path) -> None:
    from tests.test_actions_unit import PHRASED, _conv, _refer_json, _say
    from tests.test_actions_unit import StubLLM as ConvLLM

    c = _conv(ConvLLM([["国保の切り替えですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json()]), tmp_path)
    await _say(c, "国保の切り替えで聞きたいことがあって")
    await _say(c, "お願いします")
    assert c.phase == "intake"
    out = await _say(c, "もう死にたいくらいしんどい")
    assert out[0] == "話してくださって、ありがとうございます。" and c.phase == "listen"


async def test_fixed_replies_for_other_person_opinion_ai_privacy_and_cannot() -> None:
    from src.persona import Persona

    d = Dialogue(StubLLM(["x"]), DialogueConfig(), facility=FAC, now=_NOON, persona=Persona("臨時職員"),
                 enabled_actions=("refer",))
    assert (await _collect(d, "さっきの人、何の相談だったの？"))[0].startswith("ごめんなさい、ほかの方のこと")
    assert d.last_safety == "other_person"
    out = await _collect(d, "選挙は誰に入れたらいいと思う？")
    assert d.last_safety == "opinion" and "選挙管理委員会事務局（3階）" in out[-1]
    assert "AI" in (await _collect(d, "いどちゃんはAIなの？"))[0]
    assert (await _collect(d, "録音してる？"))[0] == "声は録音していません。"
    out = await _collect(d, "お金貸してくれん？")
    assert d.last_safety == "cannot_money" and "社会福祉課（2階）" in out[-1]
    await _collect(d, "代わりに申請しといてよ")
    assert d.last_safety == "cannot_apply" and d.last_proposal == "refer"
    await _collect(d, "あの課の人、感じ悪いよね")
    assert d.last_safety == "complaint" and d.last_proposal == "refer"
    await _collect(d, "国保に入りたいんじゃけど")
    assert d.last_safety is None


async def test_staff_alert_on_crisis(tmp_path: Path) -> None:
    import httpx

    from src.config import StaffConfig
    from src.edge_server import StaffPanel, create_app
    from src.referrals import ReferralStore, VisitLog

    panel = StaffPanel(ReferralStore(tmp_path, "etajima", 30, csv_export=False), VisitLog(tmp_path, "etajima"),
                       StaffConfig(password="pw"))
    item = panel.call_staff("crisis")
    assert "深刻な打ち明け" in item["label"] and "text" not in item  # 発話の本文は持たない
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none", panel), client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        rows = (await c.get("/staff/api/alerts", auth=("staff", "pw"))).json()
        assert [r["kind"] for r in rows] == ["crisis"]
        await c.post(f"/staff/api/alerts/{item['id']}/done", auth=("staff", "pw"))
        assert (await c.get("/staff/api/alerts", auth=("staff", "pw"))).json() == []


def test_import_check_requires_page_words_and_context_of_numbers() -> None:
    dp: object = _draft_module()
    check = dp.check_value  # type: ignore[attr-defined]
    page = ["## 再交付\nマイナンバーカード再交付 1件 800円\n## 更新\n暗証番号が分かるようにしてお越しください。"]
    words = ["マイナンバーカード", "更新"]
    assert check("暗証番号が分かるようにしてお越しください。", page) == []
    assert check("印鑑をお持ちください。", page)[0].startswith("語「印鑑」")
    # 再交付の金額を更新の金額にしない（数の前後に「更新」が無い）
    assert check("マイナンバーカードの更新の手数料は800円です。", page, words, ["更新"])
    assert check("マイナンバーカード再交付 1件 800円", page, words) == []
    assert check("マイナンバーカード再交付 1件 900円", page, words)[0].startswith("数・語「900円」")
    assert check("3歳未満は15,000円", ["| 3歳未満 | 15,000円 |"], ["児童手当"]) == []  # 桁区切りをそろえる


async def test_repeated_question_gets_the_same_answer_again() -> None:
    llm = StubLLM(["国保の手続きは保健医療課で承ります。"])
    d = Dialogue(llm, DialogueConfig(), facility=FAC, now=_NOON)
    first = await _collect(d, "国保の手続きはどこですか")
    again = await _collect(d, "国保の手続きはどこですか")
    assert first == again == ["国保の手続きは保健医療課で承ります。"]


# ---- C の誤りの型ごとの安全装置 ------------------------------------------------


async def test_floor_and_place_of_departments_are_fixed() -> None:
    d = Dialogue(StubLLM(["期日前投票は1階ロビーにある選挙管理委員会事務局で受付しています。",
                          "担当は農業委員会事務局で、場所は4階です。", "子育て支援課（2階）で詳しくご説明できます。"]),
                 DialogueConfig(), facility=FAC, now=_NOON)
    out = "".join(await _collect(d, "どこで聞けますか"))
    assert "3階にある選挙管理委員会事務局" in out and "場所は3階" in out
    assert "子育て支援課（この建物ではなく にこ♡にこハウス" in out and "（2階）" not in out


async def test_invented_documents_are_dropped() -> None:
    d = Dialogue(StubLLM(["婚姻届を出すには、戸籍謄本や住民票の写しが必要です。", "市民課で受け付けています。"]),
                 DialogueConfig(), facility=FAC, now=_NOON)
    out = "".join(await _collect(d, "婚姻届に何がいりますか"))
    assert out.startswith("婚姻届を出すには、" + DialogueConfig().bring_caution) and "戸籍謄本" not in out


async def test_stale_future_and_stage_direction_are_dropped() -> None:
    d = Dialogue(StubLLM(["2025年11月1日から始まるサービスです。", "それはいいですね。（相手が話すのを待つ）", "図書館でどうぞ。"]),
                 DialogueConfig(), facility=FAC, now=_NOON)
    out = "".join(await _collect(d, "図書館のカードを作りたい"))
    assert "2025年" not in out and "待つ" not in out


async def test_department_guess_is_dropped_when_rule_cannot_decide(tmp_path: Path) -> None:
    d0 = _book(tmp_path, _rec(id="iju", name="移住・定住の支援", aliases=["移住"], department="", fee=None, faq=[]))
    procs = load_procedures(SERVER_ROOT / "context", d0)
    d = Dialogue(StubLLM(["移住の支援は、社会福祉課（2階）でご案内します。", "補助金の制度があります。"]),
                 DialogueConfig(), facility=FAC, procedures=procs, now=_NOON)
    out = await _collect(d, "移住の支援はありますか")
    assert not any("社会福祉課" in x for x in out)
    assert out[-1] == "手続きの詳しいことは、担当の課でご案内します。"


async def test_residence_question_with_broken_honorific_is_dropped() -> None:
    d = Dialogue(StubLLM(["ゴミの日は地域によって異なります。", "ご住まいの地域を教えていただけますか？"]),
                 DialogueConfig(), facility=FAC, now=_NOON)
    assert "".join(await _collect(d, "ゴミの日っていつ")) == "ゴミの日は地域によって異なります。"


async def test_unknown_department_name_is_fixed_or_dropped() -> None:
    procs = load_procedures(SERVER_ROOT / "context")
    d = Dialogue(StubLLM(["転入届は住民課（1階）でご相談ください。"]), DialogueConfig(), facility=FAC,
                 procedures=procs, now=_NOON)
    assert "".join(await _collect(d, "転入の届けを出したいのですが")) == "転入届は市民課（1階）でご相談ください。"
    d2 = Dialogue(StubLLM(["高齢者医療課でご案内します。", "制度があります。"]), DialogueConfig(), facility=FAC, now=_NOON)
    assert await _collect(d2, "何か制度はありますか") == ["制度があります。"]


async def test_place_follows_corrected_department() -> None:
    procs = load_procedures(SERVER_ROOT / "context")
    d = Dialogue(StubLLM(["まずは子育て支援課（にこ♡にこハウス）でご相談くださいね。"]), DialogueConfig(),
                 facility=FAC, procedures=procs, now=_NOON)
    out = "".join(await _collect(d, "子どもの学校かわるんじゃけど何すりゃええ？"))
    assert out == "まずは学校教育課（この建物ではなく 能美市民センター本館3階（能美町中町4859番地9））でご相談くださいね。"


async def test_date_not_in_references_is_dropped() -> None:
    d = Dialogue(StubLLM(["中央地区は10月24日がごみの日です。"]), DialogueConfig(), facility=FAC, now=_NOON)
    out = await _collect(d, "ゴミの日っていつ")
    assert "10月24日" not in "".join(out)
    d2 = Dialogue(StubLLM(["今日は10月21日ですね。"]), DialogueConfig(), facility=FAC, now=_NOON)
    assert await _collect(d2, "今日は何日") == ["今日は10月21日ですね。"]  # 【今日】にある日付は言ってよい


async def test_floor_claim_after_department_outside_building_is_dropped() -> None:
    d = Dialogue(StubLLM(["児童手当は子育て支援課でご案内できます。", "2階にありますので、お気軽にお越しください。"]),
                 DialogueConfig(), facility=FAC, now=_NOON)
    out = await _collect(d, "児童手当の手続きってどこ？")
    assert not any("2階" in x for x in out)
    d2 = Dialogue(StubLLM(["国保は保健医療課です。", "1階にありますので、どうぞ。"]), DialogueConfig(), facility=FAC, now=_NOON)
    assert "2階にあり" in "".join(await _collect(d2, "国保はどこ"))  # この建物の課なら階を直す（従来どおり）
