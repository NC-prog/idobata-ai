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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteWord:
    """拠点の語 1 つ。"""

    surface: str
    # カタカナの読み。None なら ASR のバイアシング専用
    reading: str | None = None
    # アクセント核の位置。None なら VOICEVOX 登録時に既定値を使う
    accent: int | None = None


def load_site_words(site: str, context_dir: Path) -> list[SiteWord]:
    """拠点の YAML から語を読む（表記の重複は先勝ち）。

    ファイルが無い・壊れている場合は空リストを返す（ASR・TTS は語彙なしで動く）。
    """
    path = context_dir / f"{site}.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("拠点コンテキストがありません: %s", path)
        return []
    except yaml.YAMLError:
        logger.exception("拠点コンテキストの読み込みに失敗: %s", path)
        return []
    if not isinstance(data, dict):
        return []

    words: list[SiteWord] = []
    seen: set[str] = set()
    for value in data.values():
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
