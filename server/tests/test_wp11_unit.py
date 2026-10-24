"""WP11: お知らせ（選ぶ順番・出さない記事・画像を隠す・職員パネルの要約の安全装置・QR）。"""

from __future__ import annotations

import base64
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
