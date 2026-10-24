"""職員の要約を「確認済みとして保存」（WP13）: 日付・期限・金額・支払方法の本文照合だけを外し、ほかの検査と長さは残す。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DialogueConfig, NewsConfig
from src.news_panel import NewsPanel, SummaryRejected
from tests.test_wp11_unit import TODAY, _book, _checker, _rec

# 120 字ほどの要約。「11月20日」は本文に無い（職員が記事の PDF などで確かめた、という場面）
VERIFIED = ("町道の津浪巡回線は、11月10日から11月14日まで終日全面通行止めです。工事の都合で、11月20日まで延びることがあります。"
            "通行止めの間は、迂回路の案内看板に沿ってお進みください。お出かけの際は、時間に余裕を持ってお越しください。")


def _panel(tmp_path: Path, sec: float = 18.0) -> NewsPanel:
    book = _book(tmp_path, _rec("a"))
    return NewsPanel(book, _checker, DialogueConfig(), NewsConfig(), today=lambda: TODAY, speech_seconds=lambda _t: sec)


def test_verified_save_skips_only_body_grounding(tmp_path: Path) -> None:
    assert 110 <= len(VERIFIED) <= 140
    panel = _panel(tmp_path)
    with pytest.raises(SummaryRejected) as e:
        panel.edit_summary("a", VERIFIED, "staff")  # 「要約を保存」は今までどおり全部の検査
    assert "本文に無い数・日付・支払方法" in e.value.reasons
    d = panel.edit_summary("a", VERIFIED, "staff", verified=True)
    assert d["summary"] == VERIFIED and d["verified_by"] == "staff" and d["verified_at"]
    assert d["summary_checks"]["by"] == "staff_verified" and d["summary_checks"]["speech_sec"] == 18.0
    assert "本文に無い数・日付・支払方法" in d["summary_checks"]["skipped"]
    row = panel.row(panel.book.get("a"), TODAY)  # type: ignore[arg-type]
    assert row["verified"] and not row["verified_stale"]


@pytest.mark.parametrize(("bad", "reason"), [
    (VERIFIED.replace("迂回路の案内看板", "road closed の看板"), "英字: road・closed（大文字の略語は 3 語まで、ほかは allowed_terms.yaml）"),
    (VERIFIED.replace("お出かけの際は", "詳しくは住民課へ。お出かけの際は"), "拠点に無い課の名前"),
    ("町道の津浪巡回線は、11月10日から11月14日まで終日全面通行止めです。", "長さ（37 字。110〜140 字）"),
])
def test_verified_save_keeps_other_checks(tmp_path: Path, bad: str, reason: str) -> None:
    with pytest.raises(SummaryRejected) as e:
        _panel(tmp_path).edit_summary("a", bad, "staff", verified=True)
    assert reason in e.value.reasons


def test_verified_save_checks_speech_seconds(tmp_path: Path) -> None:
    with pytest.raises(SummaryRejected) as e:
        _panel(tmp_path, sec=25.0).edit_summary("a", VERIFIED, "staff", verified=True)
    assert "読み上げ 25.0 秒（15〜22 秒）" in e.value.reasons


def test_plain_save_clears_verified_and_stale_mark(tmp_path: Path) -> None:
    panel = _panel(tmp_path)
    panel.edit_summary("a", VERIFIED, "staff", verified=True)
    r = panel.book.get("a")
    assert r is not None
    r.content_hash = "changed"  # 夜に本文が変わった
    assert r.verified_stale
    ok = "町道の津浪巡回線は、11月10日から11月14日まで終日全面通行止めです。迂回路をご利用ください。"
    d = panel.edit_summary("a", ok, "staff2")  # ふつうに直したら確認済みの印は外れる
    assert d["verified_by"] == "" and d["edited_by_staff"]
