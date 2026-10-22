"""テーマ（拠点×季節のアバター・背景・切り出しの季節の一言。``server/context/themes.yaml``、WP08）。

暦（月日）で自動で選び、職員パネルから手動でも切り替えられる。どのテーマを今エッジに出しているか（applied）と、
次に出したいテーマ（wanted）を分けて持ち、会話中は切り替えない（main が輪の無いときに apply する）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 配信してよいファイル名（ディレクトリを含まない・隠しファイルでない）
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:vrm|jpg|jpeg|png|webp)$")


def safe_name(name: str) -> bool:
    """/edge/avatars/・/edge/backgrounds/ で配信してよいファイル名か。"""
    return bool(_SAFE_NAME.match(name)) and ".." not in name


@dataclass(frozen=True)
class Theme:
    name: str
    label: str
    avatar: str
    background: str
    opener_line: str = ""


@dataclass(frozen=True)
class CalendarRule:
    theme: str
    start: tuple[int, int]  # (月, 日)
    end: tuple[int, int]

    def matches(self, d: date) -> bool:
        md = (d.month, d.day)
        if self.start <= self.end:
            return self.start <= md <= self.end
        return md >= self.start or md <= self.end  # 年をまたぐ（12-01〜02-29）


@dataclass(frozen=True)
class SiteThemes:
    themes: dict[str, Theme]
    calendar: tuple[CalendarRule, ...]

    def auto(self, d: date) -> Theme | None:
        """暦で選ぶテーマ（上から順に見て最初に当たったもの）。"""
        for rule in self.calendar:
            if rule.matches(d) and rule.theme in self.themes:
                return self.themes[rule.theme]
        return next(iter(self.themes.values()), None)


def _month_day(value: Any) -> tuple[int, int]:
    m, d = str(value).split("-")
    month, day = int(m), int(d)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"月日が不正です: {value}")
    return month, day


def load_themes(path: Path, site: str) -> SiteThemes:
    """themes.yaml から拠点のテーマを読む。無い・壊れているときは空（従来どおりサンプルのアバターと季節の背景）。"""
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entry = (data.get("sites") or {}).get(site) or {}
        themes: dict[str, Theme] = {}
        for name, t in (entry.get("themes") or {}).items():
            if not isinstance(t, dict) or not t.get("avatar") or not t.get("background"):
                logger.warning("テーマ %s/%s に avatar か background がありません", site, name)
                continue
            for key in ("avatar", "background"):
                if not safe_name(str(t[key])):
                    raise ValueError(f"テーマ {name} の {key} のファイル名が不正です: {t[key]}")
            themes[str(name)] = Theme(str(name), str(t.get("label") or name), str(t["avatar"]),
                                      str(t["background"]), str(t.get("opener_line") or ""))
        rules = tuple(CalendarRule(str(r["theme"]), _month_day(r["from"]), _month_day(r["to"]))
                      for r in entry.get("calendar") or [])
    except (OSError, yaml.YAMLError, ValueError, KeyError, AttributeError):
        logger.warning("テーマを読めません: %s", path, exc_info=True)
        return SiteThemes({}, ())
    for rule in rules:
        if rule.theme not in themes:
            logger.warning("テーマの暦に未定義のテーマがあります: %s", rule.theme)
    return SiteThemes(themes, rules)


class ThemeSelector:
    """自動（暦）と手動（職員パネル）の選択と、今エッジに出しているテーマ。"""

    def __init__(self, site_themes: SiteThemes) -> None:
        self.site_themes = site_themes
        self.override: str | None = None
        self.applied: Theme | None = None

    @property
    def choices(self) -> list[Theme]:
        return list(self.site_themes.themes.values())

    def wanted(self, d: date) -> Theme | None:
        """今出したいテーマ（手動が優先）。"""
        if self.override is not None and self.override in self.site_themes.themes:
            return self.site_themes.themes[self.override]
        return self.site_themes.auto(d)

    def set_override(self, name: str | None) -> None:
        """手動で選ぶ（None で自動に戻す）。未定義の名前は ValueError。"""
        if name is not None and name not in self.site_themes.themes:
            raise ValueError(f"未定義のテーマ: {name}")
        self.override = name
        logger.info("テーマを%sにしました: %s", "手動" if name else "自動", name or "-")

    def status(self, d: date) -> dict[str, Any]:
        """職員パネルに出す状態。"""
        auto = self.site_themes.auto(d)
        wanted = self.wanted(d)
        return {
            "choices": [{"name": t.name, "label": t.label} for t in self.choices],
            "auto": auto.name if auto else None,
            "override": self.override,
            "wanted": wanted.name if wanted else None,
            "applied": self.applied.name if self.applied else None,
        }
