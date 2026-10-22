"""能動的な切り出し・沈黙・別れの定型文（``server/context/openers.yaml``、WP02 C）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Openers:
    greeting: str = "{greeting}。"
    weather: tuple[tuple[str, str], ...] = ()  # (天気の語, 一言)
    cold: tuple[float, str] | None = None  # (この気温未満, 一言)
    hot: tuple[float, str] | None = None  # (この気温より上, 一言)
    questions: tuple[str, ...] = ()
    silence: tuple[str, ...] = ()
    farewell: tuple[str, ...] = field(default_factory=tuple)


def load_openers(path: Path) -> Openers:
    """openers.yaml を読む。無い・壊れているときは空（切り出し・沈黙・別れの一言を言わない）。"""
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("切り出しの定型文を読めません: %s", path, exc_info=True)
        return Openers()
    op = data.get("opener") or {}
    cold, hot = op.get("cold") or {}, op.get("hot") or {}
    return Openers(
        greeting=str(op.get("greeting") or "{greeting}。"),
        weather=tuple((str(w["when"]), str(w["text"])) for w in op.get("weather") or []
                      if isinstance(w, dict) and w.get("when") and w.get("text")),
        cold=(float(cold["below"]), str(cold["text"])) if cold.get("text") else None,
        hot=(float(hot["above"]), str(hot["text"])) if hot.get("text") else None,
        questions=tuple(str(q) for q in op.get("questions") or []),
        silence=tuple(str(x) for x in data.get("silence") or []),
        farewell=tuple(str(x) for x in data.get("farewell") or []),
    )


def opener_lines(op: Openers, greeting: str, weather: str | None, temp: float | None, n: int,
                 season_line: str = "") -> list[str]:
    """切り出しの文: 時間帯の挨拶 ＋ 季節・天気の一言（あれば）＋ 開いた問い 1 つ（n 回目の会話で回す）。

    天気の一言は、寒暖が際立っていれば気温の一言、そうでなければ今日の予報の最初の天気（「晴れ　後　くもり」なら晴れ）。
    テーマの季節の一言（WP08、themes.yaml の opener_line）は、寒暖の一言が無いとき、天気の一言と 1 会話おきに入れ替える
    （天気が分からなければ毎回）。
    """
    lines = [op.greeting.format(greeting=greeting)]
    line = None
    if temp is not None and op.cold is not None and temp < op.cold[0]:
        line = op.cold[1]
    elif temp is not None and op.hot is not None and temp > op.hot[0]:
        line = op.hot[1]
    else:
        if weather and weather.strip():
            head = weather.replace("\u3000", " ").split()[0]
            line = next((text for word, text in op.weather if word in head), None)
        if season_line and (line is None or n % 2 == 1):
            line = season_line
    if line:
        lines.append(line)
    if op.questions:
        lines.append(op.questions[n % len(op.questions)])
    return lines


def all_lines(op: Openers, greetings: list[str]) -> list[str]:
    """事前に合成しておく文（切り出し・沈黙・別れの全部。決まった文だけなので起動時に合成できる）。"""
    out = [op.greeting.format(greeting=g) for g in greetings]
    out += [t for _, t in op.weather] + [x[1] for x in (op.cold, op.hot) if x is not None]
    out += [*op.questions, *op.silence, *op.farewell]
    return list(dict.fromkeys(out))

