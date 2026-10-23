"""WP10: 実写の背景のクレジット（credits.json）と、仮置きに落ちるときの動き。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import ThemeConfig
from src.themes import BackgroundCredit, background_credit, load_themes
from tests.test_wp08_unit import THEMES, _loop, _ThemeEdge

FMT = ThemeConfig().credit_format


def _credits(d: Path, **entries: dict[str, Any]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "credits.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def test_credit_only_for_real_photo_listed_in_credits(tmp_path: Path) -> None:
    bg = tmp_path / "backgrounds"
    _credits(bg, akiota_autumn={"file": "akiota_autumn.jpg", "author": "HKT3012", "license": "CC BY-SA 4.0",
                                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/deed.ja",
                                "credit_required": True, "anchor": "center"},
             akiota_summer={"file": "akiota_summer.jpg", "author": "かるちる", "license": "CC0",
                            "credit_required": False, "anchor": "left"},
             etajima_winter={"file": "etajima_winter.jpg", "author": "X", "license": "CC BY 3.0",
                             "credit_required": True, "anchor": "sideways"})
    for name in ("akiota_autumn.jpg", "akiota_summer.jpg", "etajima_winter.jpg", "akiota_spring.jpg"):
        (bg / name).write_bytes(b"jpg")
    got = background_credit(bg, "akiota_autumn.jpg", "credits.json", FMT)
    assert got.text == ("写真: HKT3012 / CC BY-SA 4.0（creativecommons.org/licenses/by-sa/4.0）/ Wikimedia Commons"
                        "（切り抜き・加工）") and got.anchor == "center"
    # CC0・パブリックドメインは文言なし（出所は credits.json に残る）。見せる位置は使う
    assert background_credit(bg, "akiota_summer.jpg", "credits.json", FMT) == BackgroundCredit("", "left")
    # 知らない位置は中央に
    assert background_credit(bg, "etajima_winter.jpg", "credits.json", FMT).anchor == "center"
    # 実写はあるが credits.json に無い → 出さない
    assert background_credit(bg, "akiota_spring.jpg", "credits.json", FMT) == BackgroundCredit()
    # 実写が無い（エッジの仮置きのイラストに落ちる）→ credits.json にあっても出さない
    (bg / "akiota_autumn.jpg").unlink()
    assert background_credit(bg, "akiota_autumn.jpg", "credits.json", FMT) == BackgroundCredit()
    # credits.json が壊れていても落ちない
    (bg / "credits.json").write_text("{", encoding="utf-8")
    assert background_credit(bg, "akiota_summer.jpg", "credits.json", FMT) == BackgroundCredit()
    assert background_credit(tmp_path / "none", "x.jpg", "credits.json", FMT) == BackgroundCredit()


async def test_theme_message_carries_credit_and_falls_back(tmp_path: Path, monkeypatch: Any) -> None:
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "config", replace(main_mod.config, data_dir=tmp_path, theme=ThemeConfig()))
    bg = tmp_path / "backgrounds"
    _credits(bg, etajima_autumn={"file": "etajima_autumn.jpg", "author": "柳田亮", "license": "CC BY-SA 3.0",
                                 "credit_required": True, "anchor": "right"})
    (bg / "etajima_autumn.jpg").write_bytes(b"jpg")
    app, _ = _loop()
    edge = _ThemeEdge()
    app._edge = edge  # type: ignore[assignment]
    app._themes.site_themes = load_themes(THEMES, "etajima")
    for name, expect in (("festival", ("写真: 柳田亮 / CC BY-SA 3.0", "right")),  # お祭りは秋の背景
                         ("spring", ("", "center"))):  # 実写の無い季節は仮置き・文言なし
        await app._send_theme(app._themes.site_themes.themes[name])
        credit, anchor = edge.credits[-1]
        assert credit.startswith(expect[0]) and anchor == expect[1]
        assert edge.themes[-1][2] == f"backgrounds/etajima_{'autumn' if name == 'festival' else name}.jpg"
    await app._tts.close()
