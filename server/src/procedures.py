"""手続きの別名表（``server/context/procedures.yaml``）と手続きナレッジから、担当課を規則で決めること（WP04・WP09）。

担当課の決定（来訪者に課を聞き返さない・LLM に課を推測させない）:
- 対話で使える（承認済みの）手続きナレッジの name・aliases が発話にあれば、その department（WP09）
- 残りは procedures.yaml の別名表の語（term・aliases）と拠点 YAML の ``facility.departments[].handles`` を突き合わせる
  - 発話に term か alias があれば、その term を言ったものとする（「国保」→ 国民健康保険）
  - handles の語（「・」で分けた各語も）が発話にそのまま出ていれば、それも言ったものとする
  - 言った語のうち、ほかの言った語に含まれるものは捨てる（「国保税」は国民健康保険税で、国民健康保険ではない）
  - 残った語を handles に持つ課が候補
- 候補が一意でなければ「未特定」（職員が職員パネルで付ける）

紹介状の聞き取り（intake）と持ち物（bring）は、承認済みの手続きナレッジから引く（procedures.yaml からは移した）。
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
from .procedure_kb import IntakeSet, ProcedureBook, ProcedureRecord
from .site_context import Facility, SiteWord

logger = logging.getLogger(__name__)

# actions.py などは IntakeSet をここから使う
__all__ = ["AsrFixes", "IntakeSet", "Procedure", "Procedures", "load_procedures"]

PROCEDURES_FILE = "procedures.yaml"
# 職員パネルから足した聞き取り間違いの対応表（データディレクトリ。procedures.yaml の asr_fixes に足す）
ASR_FIXES_FILE = "asr_fixes.jsonl"
ASR_FIX_MAX_CHARS = 20

# handles の語の括弧書き（「国民健康保険（資格・給付）」）は突き合わせに使わない
_PAREN_RE = re.compile(r"[（(].*?[）)]")


@dataclass(frozen=True)
class Procedure:
    term: str
    aliases: tuple[str, ...] = ()
    # 紹介状に付けるタグ（tags.yaml の統制語彙。手続きが分かったときは LLM の選んだタグより先に使う）
    tags: tuple[str, ...] = ()
    # handles を持たない課で、名前で担当が分かるもの（WP09）
    departments: tuple[str, ...] = ()


@dataclass(frozen=True)
class Procedures:
    items: tuple[Procedure, ...] = ()
    readings: tuple[SiteWord, ...] = ()
    asr_exclude: frozenset[str] = frozenset()
    # 手続きナレッジ（WP09。~/idobata-data/procedures/<site>/。無ければ別名表だけで動く）
    book: ProcedureBook | None = None

    def words(self) -> list[SiteWord]:
        """ASR の語彙と VOICEVOX の辞書に渡す語（読みのあるものだけ辞書に入る）。"""
        reading = {w.surface: w for w in self.readings}
        out: list[SiteWord] = []
        seen: set[str] = set()
        for p in self.items:
            for s in (p.term, *p.aliases):
                if s in self.asr_exclude or s in seen:
                    continue
                seen.add(s)
                out.append(reading.pop(s, SiteWord(s)))
        # 読みだけ持ち、ASR からは外した語も辞書には入れる
        out += [SiteWord(w.surface, w.reading, w.accent, asr=False) for w in reading.values()]
        return out

    def surfaces(self) -> tuple[str, ...]:
        """別名表の語（手続きナレッジの突き合わせで、長い語に含まれる当たりを捨てるのに使う）。"""
        return tuple(s for p in self.items for s in (p.term, *p.aliases))

    def matched(self, text: str) -> list[ProcedureRecord]:
        """発話で言った手続きナレッジ（全状態。出てきた順）。"""
        return self.book.match(text) if self.book is not None else []

    def usable_matched(self, text: str) -> list[ProcedureRecord]:
        """発話で言った、対話で使える（承認済みの）手続きナレッジ。"""
        book = self.book
        return [r for r in self.matched(text) if book is not None and book.usable(r)]

    def _rest_text(self, text: str, used: list[ProcedureRecord]) -> str:
        """手続きナレッジで当たった語を伏せた発話（残りを別名表で見る）。"""
        for r in used:
            for s in sorted(r.surfaces, key=len, reverse=True):
                text = text.replace(s, "＿")
        return text

    def mentioned(self, text: str, facility: Facility | None) -> set[str]:
        """発話で言った手続きの語（別名表の term と handles の語）。ほかの語に含まれるものは除く。"""
        found = {p.term for p in self.items if any(s in text for s in (p.term, *p.aliases))}
        found |= {k for k in _handle_keywords(facility) if k in text}
        return {k for k in found if not any(k != o and k in o for o in found)}

    def intake_for(self, text: str, facility: Facility | None) -> IntakeSet | None:
        """発話の手続きに合う聞き取りの組（承認済みの手続きナレッジのうち、発話で先に出たもの）。"""
        del facility
        for r in self.usable_matched(text):
            if r.intake or r.bring:
                return r.intake_set()
        return None

    def tags_for(self, text: str, facility: Facility | None) -> list[str]:
        """発話で言った手続きのタグ（手続きナレッジ → 別名表の順、重複なし）。"""
        used = self.usable_matched(text)
        said = self.mentioned(self._rest_text(text, used), facility)
        tags = [t for r in used for t in r.tags]
        tags += [t for p in self.items if p.term in said for t in p.tags]
        return list(dict.fromkeys(tags))

    def departments_for(self, text: str, facility: Facility | None) -> list[str]:
        """発話から決まる担当課の候補（facility の並び順）。"""
        if facility is None:
            return []
        names = [d.name for d in facility.departments]
        used = self.usable_matched(text)
        found = {r.department for r in used if r.department in names}
        said = self.mentioned(self._rest_text(text, used), facility)
        found |= {d.name for d in facility.departments if any(k in said for k in _department_keywords(d.handles))}
        found |= {n for p in self.items if p.term in said for n in p.departments if n in names}
        out = [n for n in names if n in found]
        if len(out) > 1:
            # 候補にこの建物の課が 1 つだけなら、その課（安芸太田の戸籍: 税務住民課・加計支所・筒賀支所 → 税務住民課）
            here = [d.name for d in facility.departments if d.name in out and not d.place]
            if len(here) == 1:
                return here
        return out


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


def load_procedures(context_dir: Path, knowledge_dir: Path | None = None, keep_recheck: bool = True) -> Procedures:
    """procedures.yaml（別名表）と手続きナレッジ（knowledge_dir）を読む。

    procedures.yaml が無い・壊れているときは別名表が空（規則では課が決まらず、未特定になる）。
    knowledge_dir が None なら手続きナレッジなし（ASR の語彙を作るときなど）。
    """
    path = context_dir / PROCEDURES_FILE
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("手続きの語の表を読めません: %s", path, exc_info=True)
        data = {}
    items = tuple(
        Procedure(str(p["term"]), _strs(p.get("aliases")), _strs(p.get("tags")), _strs(p.get("departments")))
        for p in data.get("procedures") or [] if isinstance(p, dict) and p.get("term")
    )
    readings = tuple(
        SiteWord(str(r["surface"]), str(r["reading"]), r.get("accent") if isinstance(r.get("accent"), int) else None)
        for r in data.get("readings") or [] if isinstance(r, dict) and r.get("surface") and r.get("reading")
    )
    book = None
    if knowledge_dir is not None:
        book = ProcedureBook(knowledge_dir, keep_recheck,
                             tuple(s for p in items for s in (p.term, *p.aliases)))
        n = len(book.records())
        logger.info("手続きナレッジ: %s（%d 件、うち対話で使えるもの %d 件）", knowledge_dir, n,
                    sum(1 for r in book.records() if book.usable(r)))
    return Procedures(items, readings, frozenset(str(x) for x in data.get("asr_exclude") or []), book)


def _strs(value: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in value or [])


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
