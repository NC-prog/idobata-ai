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

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .knowledge import JST
from .site_context import Facility, SiteWord

logger = logging.getLogger(__name__)

PROCEDURES_FILE = "procedures.yaml"
# 職員パネルから足した聞き取り間違いの対応表（データディレクトリ。procedures.yaml の asr_fixes に足す）
ASR_FIXES_FILE = "asr_fixes.jsonl"
ASR_FIX_MAX_CHARS = 20

# handles の語の括弧書き（「国民健康保険（資格・給付）」）は突き合わせに使わない
_PAREN_RE = re.compile(r"[（(].*?[）)]")


@dataclass(frozen=True)
class BringItem:
    """持ち物の案内 1 つ。when の語が聞き取りの答えか発話にあれば言う（空なら常に）。"""

    item: str
    when: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntakeSet:
    """紹介状の同意の後に伺う項目の組。when の語が発話にあれば使う（空なら常に）。"""

    questions: tuple[str, ...]
    when: tuple[str, ...] = ()
    bring: tuple[BringItem, ...] = ()

    def bring_for(self, text: str) -> str | None:
        """聞き取りの答えと発話から分かる持ち物（最初に合った 1 つ）。"""
        for b in self.bring:
            if not b.when or any(w in text for w in b.when):
                return b.item
        return None


@dataclass(frozen=True)
class Procedure:
    term: str
    aliases: tuple[str, ...] = ()
    intake: tuple[IntakeSet, ...] = ()
    # 紹介状に付けるタグ（tags.yaml の統制語彙。手続きが分かったときは LLM の選んだタグより先に使う）
    tags: tuple[str, ...] = ()


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

    def intake_for(self, text: str, facility: Facility | None) -> IntakeSet | None:
        """発話の手続きに合う聞き取りの組（表の順で最初のもの）。"""
        said = self.mentioned(text, facility)
        for p in self.items:
            if p.term not in said:
                continue
            for s in p.intake:
                if not s.when or any(w in text for w in s.when):
                    return s
        return None

    def tags_for(self, text: str, facility: Facility | None) -> list[str]:
        """発話で言った手続きのタグ（表の順、重複なし）。"""
        said = self.mentioned(text, facility)
        return list(dict.fromkeys(t for p in self.items if p.term in said for t in p.tags))

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
        Procedure(str(p["term"]), _strs(p.get("aliases")),
                  tuple(_intake(x) for x in p.get("intake") or [] if isinstance(x, dict) and x.get("questions")),
                  _strs(p.get("tags")))
        for p in data.get("procedures") or [] if isinstance(p, dict) and p.get("term")
    )
    readings = tuple(
        SiteWord(str(r["surface"]), str(r["reading"]), r.get("accent") if isinstance(r.get("accent"), int) else None)
        for r in data.get("readings") or [] if isinstance(r, dict) and r.get("surface") and r.get("reading")
    )
    return Procedures(items, readings, frozenset(str(x) for x in data.get("asr_exclude") or []))


def _strs(value: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in value or [])


def _intake(data: dict[str, Any]) -> IntakeSet:
    bring = tuple(BringItem(str(b["item"]), _strs(b.get("when")))
                  for b in data.get("bring") or [] if isinstance(b, dict) and b.get("item"))
    return IntakeSet(_strs(data.get("questions")), _strs(data.get("when")), bring)


class AsrFixes:
    """聞き取り間違いの対応表（「国宝」→「国保」）。ASR の出力を対話と課の照合に渡す前に直す。

    procedures.yaml の asr_fixes と、職員パネルから足した行（データディレクトリの asr_fixes.jsonl）を合わせる。
    """

    def __init__(self, context_dir: Path, data_dir: Path) -> None:
        self.path = data_dir / ASR_FIXES_FILE
        self._base: dict[str, str] = {}
        try:
            data: Any = yaml.safe_load((context_dir / PROCEDURES_FILE).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        for f in data.get("asr_fixes") or []:
            if isinstance(f, dict) and f.get("surface") and f.get("correct"):
                self._base[str(f["surface"])] = str(f["correct"])
        self._added: dict[str, str] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    f = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(f, dict) and f.get("surface") and f.get("correct"):
                    self._added[str(f["surface"])] = str(f["correct"])

    @property
    def table(self) -> dict[str, str]:
        return {**self._base, **self._added}

    def entries(self) -> list[dict[str, str]]:
        """職員パネルの一覧（どこで定義したかも付ける）。"""
        return ([{"surface": k, "correct": v, "source": "procedures.yaml"} for k, v in self._base.items()]
                + [{"surface": k, "correct": v, "source": "職員パネル"} for k, v in self._added.items()])

    def apply(self, text: str) -> str:
        """長い表記から順に置き換える。"""
        table = self.table
        out = text
        for surface in sorted(table, key=len, reverse=True):
            out = out.replace(surface, table[surface])
        if out != text:
            logger.info("聞き取り間違いの対応表で直しました")
        return out

    def add(self, surface: str, correct: str) -> dict[str, str]:
        surface, correct = surface.strip(), correct.strip()
        if not (0 < len(surface) <= ASR_FIX_MAX_CHARS and 0 < len(correct) <= ASR_FIX_MAX_CHARS):
            raise ValueError(f"表記は 1〜{ASR_FIX_MAX_CHARS} 文字にしてください")
        if surface == correct or re.search(r"[\r\n]", surface + correct):
            raise ValueError("聞き取り間違いと正しい表記が同じか、改行を含んでいます")
        item = {"surface": surface, "correct": correct,
                "added_at": datetime.now(JST).isoformat(timespec="seconds")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._added[surface] = correct
        logger.info("聞き取り間違いの対応表に 1 行足しました")
        return item
