"""site_context（拠点 YAML の語と読み）のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import SERVER_ROOT
from src.site_context import SiteWord, load_site_words

# 作業指示（WP01 A-2）で最低限登録する語
_REQUIRED = {
    "etajima": {"江田島": "エタジマ", "能美": "ノウミ", "大柿": "オオガキ", "沖美": "オキミ",
                "小用": "コヨウ", "切串": "キリクシ", "中町": "ナカマチ"},
    "akiota": {"安芸太田": "アキオオタ", "加計": "カケ", "戸河内": "トゴウチ", "筒賀": "ツツガ",
               "三段峡": "サンダンキョウ", "温井": "ヌクイ"},
}


@pytest.mark.parametrize("site", ["etajima", "akiota"])
def test_bundled_sites_have_required_readings(site: str) -> None:
    words = {w.surface: w for w in load_site_words(site, SERVER_ROOT / "context")}
    for surface, reading in _REQUIRED[site].items():
        assert words[surface].reading == reading
    # 読みはカタカナだけ
    for w in words.values():
        if w.reading:
            assert all("゠" <= ch <= "ヿ" for ch in w.reading), w


def test_mixed_items_and_dedup(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text(
        "site_name: テスト市\n"
        "place_names:\n"
        "  - {surface: 能美, reading: ノウミ, accent: 0}\n"
        "  - {surface: 小用, reading: コヨウ}\n"
        "  - {reading: ナシ}\n"
        "offices:\n  - 能美\n  - 市役所\n",
        encoding="utf-8",
    )
    assert load_site_words("x", tmp_path) == [
        SiteWord("テスト市"),
        SiteWord("能美", "ノウミ", 0),
        SiteWord("小用", "コヨウ", None),
        SiteWord("市役所"),
    ]


def test_missing_and_broken(tmp_path: Path) -> None:
    assert load_site_words("nowhere", tmp_path) == []
    (tmp_path / "bad.yaml").write_text("a: [", encoding="utf-8")
    assert load_site_words("bad", tmp_path) == []
