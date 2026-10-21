"""手続きの常用語（``server/context/procedures.yaml``）と、担当課を規則で決めること（WP04）。

来訪者の発話から手続きの語（正式名・略称）を拾い、拠点 YAML の ``facility.departments[].handles``
と突き合わせる。推測では課を確定しない（一意に決まらなければ来訪者に聞く・確かめる）。

突き合わせ:
- 発話に term か alias があれば、その term を言ったものとする（「国保」→ 国民健康保険）
- handles の語（「・」で分けた各語も）が発話にそのまま出ていれば、それも言ったものとする
- 言った語のうち、ほかの言った語に含まれるものは捨てる（「国保税」は国民健康保険税で、国民健康保険ではない）
- 残った語を handles に持つ課が候補
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .site_context import Facility, SiteWord

logger = logging.getLogger(__name__)

PROCEDURES_FILE = "procedures.yaml"

# handles の語の括弧書き（「国民健康保険（資格・給付）」）は突き合わせに使わない
_PAREN_RE = re.compile(r"[（(].*?[）)]")


@dataclass(frozen=True)
class Procedure:
    term: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Procedures:
    items: tuple[Procedure, ...] = ()
    readings: tuple[SiteWord, ...] = ()
    asr_exclude: frozenset[str] = frozenset()

    def words(self) -> list[SiteWord]:
        """ASR の語彙と VOICEVOX の辞書に渡す語（読みのあるものだけ辞書に入る）。"""
        reading = {w.surface: w for w in self.readings}
        out: list[SiteWord] = []
        for p in self.items:
            for s in (p.term, *p.aliases):
                if s in self.asr_exclude:
                    continue
                out.append(reading.pop(s, SiteWord(s)))
        # 読みだけ持ち、ASR からは外した語も辞書には入れる
        out += [SiteWord(w.surface, w.reading, w.accent, asr=False) for w in reading.values()]
        return out

    def mentioned(self, text: str, facility: Facility | None) -> set[str]:
        """発話で言った手続きの語（term と handles の語）。ほかの語に含まれるものは除く。"""
        found = {p.term for p in self.items if any(s in text for s in (p.term, *p.aliases))}
        found |= {k for k in _handle_keywords(facility) if k in text}
        return {k for k in found if not any(k != o and k in o for o in found)}

    def departments_for(self, text: str, facility: Facility | None) -> list[str]:
        """発話から決まる担当課の候補（facility の並び順）。"""
        if facility is None:
            return []
        said = self.mentioned(text, facility)
        return [d.name for d in facility.departments
                if any(k in said for k in _department_keywords(d.handles))]


def _department_keywords(handles: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for h in handles:
        h = _PAREN_RE.sub("", h).strip()
        if h:
            out.add(h)
            out |= {x for x in h.split("・") if len(x) >= 2}
    return out


def _handle_keywords(facility: Facility | None) -> set[str]:
    out: set[str] = set()
    for d in (facility.departments if facility else ()):
        out |= _department_keywords(d.handles)
    return out


def load_procedures(context_dir: Path) -> Procedures:
    """procedures.yaml を読む。無い・壊れているときは空（規則では課が決まらず、確かめる流れになる）。"""
    path = context_dir / PROCEDURES_FILE
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("手続きの語の表を読めません: %s", path, exc_info=True)
        return Procedures()
    items = tuple(
        Procedure(str(p["term"]), tuple(str(a) for a in p.get("aliases") or []))
        for p in data.get("procedures") or [] if isinstance(p, dict) and p.get("term")
    )
    readings = tuple(
        SiteWord(str(r["surface"]), str(r["reading"]), r.get("accent") if isinstance(r.get("accent"), int) else None)
        for r in data.get("readings") or [] if isinstance(r, dict) and r.get("surface") and r.get("reading")
    )
    return Procedures(items, readings, frozenset(str(x) for x in data.get("asr_exclude") or []))
