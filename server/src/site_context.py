"""拠点コンテキスト（``server/context/<site>.yaml``）の読み込み。

YAML はカテゴリ名 → 語のリストの辞書。各語は文字列か、読みを持つ辞書で書く::

    place_names:
      - surface: 能美
        reading: ノウミ      # カタカナ。VOICEVOX のユーザー辞書に登録する
        accent: 0            # アクセント核の位置（0 = 平板）。省略時は既定値
    departments:
      - 市民課               # 読みを持たない語（ASR のバイアシングだけに使う）

同じ語彙を ASR のコンテキストバイアシング（surface）と VOICEVOX のユーザー辞書
（reading を持つ語）の両方に使う。

WP03: ``facility`` 節（拠点の固定事実: 施設名・開庁時間・課と担当・緊急連絡先）と
``allow_ascii``（応答に英字のまま出してよい語）を持てる。これらは語彙としては読まない::

    facility:
      name: 江田島市役所
      location_phrase: 江田島市役所の1階ロビー
      address_city: 江田島市
      hours: 平日 8時30分から17時15分
      departments:
        - {name: 市民課, floor: 1階, handles: [住民票, 戸籍]}
        - {name: 子育て支援課, place: 江田島町中央四丁目, handles: [児童手当]}
      emergency_lines:
        - {name: マイナンバー総合フリーダイヤル, number: 0120-95-0178}
    allow_ascii: [JR]
    outside_places: [宮島]      # 拠点の外なのに LLM が拠点のものとして挙げやすい場所
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# 語彙ではない節（拠点の固定事実など）
_NON_VOCAB_KEYS = frozenset({"facility", "allow_ascii", "outside_places"})


@dataclass(frozen=True)
class Department:
    """課 1 つ。floor（本庁舎の階）か place（別の建物）のどちらか、分かる方だけ。"""

    name: str
    floor: str | None = None
    place: str | None = None
    handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmergencyLine:
    """手続きではない緊急連絡先（知識パックに依らず案内してよい）。"""

    name: str
    number: str


@dataclass(frozen=True)
class Facility:
    """拠点の固定事実（システムプロンプトに直書きする層）。"""

    name: str
    location_phrase: str
    address_city: str = ""
    hours: str = ""
    departments: tuple[Department, ...] = ()
    emergency_lines: tuple[EmergencyLine, ...] = ()
    # 固定事実として話してよい補足（延長窓口、担当の分かれ方など）
    notes: tuple[str, ...] = ()
    allow_ascii: tuple[str, ...] = ()
    # 拠点の外なのに LLM が拠点のものとして挙げやすい場所（含む文は読み上げない）
    outside_places: tuple[str, ...] = ()


@dataclass(frozen=True)
class SiteWord:
    """拠点の語 1 つ。"""

    surface: str
    # カタカナの読み。None なら ASR のバイアシング専用
    reading: str | None = None
    # アクセント核の位置。None なら VOICEVOX 登録時に既定値を使う
    accent: int | None = None


def _load_yaml(site: str, context_dir: Path) -> dict[str, Any]:
    path = context_dir / f"{site}.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("拠点コンテキストがありません: %s", path)
        return {}
    except yaml.YAMLError:
        logger.exception("拠点コンテキストの読み込みに失敗: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def load_facility(site: str, context_dir: Path) -> Facility | None:
    """拠点の固定事実を読む。facility 節が無ければ None。"""
    data = _load_yaml(site, context_dir)
    raw = data.get("facility")
    if not isinstance(raw, dict) or not raw.get("name"):
        logger.warning("拠点 %s に facility 節がありません（固定事実なしで動きます）", site)
        return None
    departments = tuple(
        Department(
            name=str(d["name"]),
            floor=str(d["floor"]) if d.get("floor") else None,
            place=str(d["place"]) if d.get("place") else None,
            handles=tuple(str(h) for h in d.get("handles") or []),
        )
        for d in raw.get("departments") or []
        if isinstance(d, dict) and d.get("name")
    )
    lines = tuple(
        EmergencyLine(name=str(e["name"]), number=str(e["number"]))
        for e in raw.get("emergency_lines") or []
        if isinstance(e, dict) and e.get("name") and e.get("number")
    )
    return Facility(
        name=str(raw["name"]),
        location_phrase=str(raw.get("location_phrase") or raw["name"]),
        address_city=str(raw.get("address_city") or ""),
        hours=str(raw.get("hours") or ""),
        departments=departments,
        emergency_lines=lines,
        notes=tuple(str(n) for n in raw.get("notes") or [] if n),
        allow_ascii=tuple(str(a) for a in data.get("allow_ascii") or []),
        outside_places=tuple(str(a) for a in data.get("outside_places") or []),
    )


def load_site_words(site: str, context_dir: Path) -> list[SiteWord]:
    """拠点の YAML から語を読む（表記の重複は先勝ち）。

    ファイルが無い・壊れている場合は空リストを返す（ASR・TTS は語彙なしで動く）。
    facility 節の課名も語彙に含める（ASR のバイアシング用）。
    """
    data = _load_yaml(site, context_dir)
    words: list[SiteWord] = []
    seen: set[str] = set()
    facility = data.get("facility")
    dept_names = [
        d["name"] for d in (facility or {}).get("departments") or []
        if isinstance(facility, dict) and isinstance(d, dict) and d.get("name")
    ]
    values = [v for k, v in data.items() if k not in _NON_VOCAB_KEYS] + [dept_names]
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            word = _parse_item(item)
            if word is None or word.surface in seen:
                continue
            seen.add(word.surface)
            words.append(word)
    return words


def _parse_item(item: Any) -> SiteWord | None:
    if isinstance(item, str):
        return SiteWord(item) if item else None
    if not isinstance(item, dict):
        return None
    surface = item.get("surface")
    if not isinstance(surface, str) or not surface:
        logger.warning("surface の無い語を無視します: %r", item)
        return None
    reading = item.get("reading")
    accent = item.get("accent")
    return SiteWord(
        surface=surface,
        reading=reading if isinstance(reading, str) and reading else None,
        accent=accent if isinstance(accent, int) else None,
    )
