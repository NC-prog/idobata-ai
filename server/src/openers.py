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
