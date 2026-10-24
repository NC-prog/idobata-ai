"""用件の聞き分け（WP13 E）。複数の手続きが同程度に当たったら、来訪者に 1 問だけ聞く。LLM は使わない。

候補の集め方:
- 発話にある、手続きナレッジ（承認済み・下書き）と別名表の語（name・別名・extra_words）を「鍵」にする。
  ほかの鍵の中に含まれる鍵は捨てる（「国保税」と言えば「国保」は鍵にしない）。語が 1 つも無ければ、発話の名詞を鍵にする（「手当」）
- 鍵を含む語（name・別名・when・extra_words）を持つ手続きが候補（「国保」→ 国保の加入・国保の脱退・国保税）。
  別名表だけの手続き（手続きナレッジが無く、語が手続きナレッジの語と重ならないもの）も「term:<term>」として入れる
- 点: 候補を見分ける語（name を「の・・」で分けた語、別名、when、cues。鍵そのものは数えない）が発話にいくつあるか
- 1 位と 2 位の点差が margin 未満で、同点の上位が min〜max 件なら聞く（「国保に入りたい」は加入が 1 点で、聞かない）

返事の判定: 順序の語（最初・二つ目・最後…）→「ほか」→ 見分ける語の点（1 位が 1 つだけなら決まり）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ProcedureConfig
from .intents import nouns
from .procedure_kb import ProcedureRecord
from .procedures import Procedures

_PAREN_RE = re.compile(r"[（(].*?[）)]")
_SPLIT_RE = re.compile(r"[の・、（）()\s]+")
TERM_PREFIX = "term:"
OTHER = "other"


@dataclass(frozen=True)
class ClarifyRules:
    question: str = "{key}のことですね。{choices}、どちらでしょう？"
    other_choice: str = "ほかのこと"
    reask: str = "すみません、{choices}、どちらでしょう？"
    fallback: str = "手続きの詳しいことは、担当の課でご案内します。"
    other: str = "わかりました。どんなことでしょう？"
    ordinals: dict[int, tuple[str, ...]] = field(default_factory=dict)
    other_words: tuple[str, ...] = ()
    labels: dict[str, str] = field(default_factory=dict)
    cues: dict[str, tuple[str, ...]] = field(default_factory=dict)
    extra_words: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # 鍵を含んでいても、それだけでは候補に入れない語（「転入」は「転入学」の中にあるが、転校の手続きではない）
    not_join: tuple[str, ...] = ()
    # 発話の名詞を鍵にするとき、使わない一般の語
    generic_nouns: tuple[str, ...] = ()


def _words(v: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in v or [])


def load_clarify_rules(path: Path) -> ClarifyRules:
    """actions.yaml の clarify。"""
    data: Any = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("clarify") or {}
    base = ClarifyRules()
    return ClarifyRules(
        question=str(data.get("question") or base.question),
        other_choice=str(data.get("other_choice") or base.other_choice),
        reask=str(data.get("reask") or base.reask),
        fallback=str(data.get("fallback") or base.fallback),
        other=str(data.get("other") or base.other),
        ordinals={int(k): _words(v) for k, v in (data.get("ordinals") or {}).items()},
        other_words=_words(data.get("other_words")),
        labels={str(k): str(v) for k, v in (data.get("labels") or {}).items()},
        cues={str(k): _words(v) for k, v in (data.get("cues") or {}).items()},
        extra_words={str(k): _words(v) for k, v in (data.get("extra_words") or {}).items()},
        not_join=_words(data.get("not_join")),
        generic_nouns=_words(data.get("generic_nouns")),
    )


@dataclass
class Clarify:
    """聞き分けの問い（返事待ちの間、Pending に持つ）。"""

    key: str
    candidates: list[ProcedureRecord]  # 選択肢に出す順
    keys: tuple[str, ...]  # 鍵（返事の点には数えない）
    more: bool = False  # 「ほかのこと」を足したか
    reasks: int = 0

    @property
    def ids(self) -> list[str]:
        return [r.id for r in self.candidates]


class Clarifier:
    def __init__(self, procedures: Procedures, rules: ClarifyRules, cfg: ProcedureConfig) -> None:
        self._procs = procedures
        self._rules = rules
        self.rules = rules
        self._cfg = cfg

    # --- 候補 ---

    def _records(self) -> list[ProcedureRecord]:
        """手続きナレッジ（全状態）と、別名表だけの手続き。"""
        book = self._procs.book
        recs = list(book.records()) if book is not None else []
        known = {s for r in recs for s in r.surfaces}
        for p in self._procs.items:
            words = (p.term, *p.aliases)
            if not any(w in known for w in words):
                recs.append(ProcedureRecord(id=f"{TERM_PREFIX}{p.term}", name=p.term, aliases=tuple(p.aliases),
                                            status="draft"))
        return recs

    def _join_words(self, r: ProcedureRecord) -> tuple[str, ...]:
        """候補に入る語（鍵がこの中のどれかに含まれれば候補）。"""
        return (*r.surfaces, *r.when, *self._rules.extra_words.get(r.id, ()))

    def _joins(self, r: ProcedureRecord, key: str) -> bool:
        """鍵がこの手続きの語の中にあるか（not_join の語の中にあるだけなら数えない）。"""
        for w in self._join_words(r):
            if key == w:
                return True
            for nj in self._rules.not_join:
                if key in nj and key != nj:
                    w = w.replace(nj, "＿")
            if key in w:
                return True
        return False

    def _cue_words(self, r: ProcedureRecord) -> list[str]:
        """見分ける語。"""
        parts = [w.removesuffix("など") for w in _SPLIT_RE.split(r.name) if len(w.removesuffix("など")) >= 2]
        label = self._rules.labels.get(r.id, "")
        parts += [w for w in _SPLIT_RE.split(label) if len(w) >= 2 and not w.endswith(("手続き", "とき"))]
        return list(dict.fromkeys([*parts, *r.surfaces, *r.when, *self._rules.cues.get(r.id, ())]))

    def score(self, r: ProcedureRecord, text: str, keys: tuple[str, ...]) -> int:
        """見分ける語の数（鍵そのものは数えない）と、当たった鍵の数（鍵が 2 つ以上のとき。「国保の保険証」は脱退・保険証が 2）。"""
        cues = sum(1 for w in self._cue_words(r) if w in text and not any(w in k for k in keys))
        return cues + (sum(1 for k in keys if self._joins(r, k)) if len(keys) > 1 else 0)

    def find(self, text: str) -> Clarify | None:
        """聞き分けが要るなら、その問い（要らなければ None）。"""
        cfg = self._cfg
        if not cfg.clarify_enabled:
            return None
        recs = self._records()
        spans: list[tuple[int, int, str]] = []
        for r in recs:
            for w in (*r.surfaces, *self._rules.extra_words.get(r.id, ())):
                spans += [(m.start(), m.end(), w) for m in re.finditer(re.escape(w), text)]
        keys = tuple(dict.fromkeys(w for a, b, w in spans if not any(
            a2 <= a and b <= b2 and (b2 - a2) > (b - a) for a2, b2, _ in spans)))
        if not keys:
            # 手続きの語が無ければ発話の名詞（「手当」）。候補を持つ名詞がそれ 1 つだけのときに限る
            # （「子どもの学校かわる」の「子ども」は鍵にしない）。一般の語（申請・手続き・相談）は鍵にしない
            found = [n for n in nouns(text) if len(n) >= 2 and n not in self._rules.generic_nouns
                     and any(self._joins(r, n) for r in recs)]
            keys = tuple(found) if len(found) == 1 else ()
        if not keys:
            return None
        # 聞くのは、1 つの鍵に候補が 2 件以上あるとき（その語そのものがあいまい）。別々の手続きをはっきり言った
        # 発話（「子どもが生まれたので児童手当の申請」）は聞かない
        ambiguous = [k for k in keys if sum(1 for r in recs if self._joins(r, k)) >= cfg.clarify_min]
        if not ambiguous:
            return None
        family = [r for r in recs if any(self._joins(r, k) for k in ambiguous)]
        if not cfg.clarify_min <= len(family) <= cfg.clarify_max:
            return None
        scores = [self.score(r, text, keys) for r in family]
        top = max(scores)
        tied = [r for r, s in zip(family, scores, strict=True) if top - s < cfg.clarify_margin]
        if len(tied) < cfg.clarify_min:
            return None
        book = self._procs.book

        def order(r: ProcedureRecord) -> tuple[int, int]:
            exact = any(k in r.surfaces or k in self._rules.extra_words.get(r.id, ()) for k in keys)
            approved = book is not None and book.usable(r)
            return (0 if approved else 1, 0 if exact else 1)

        tied.sort(key=order)
        shown = tied[: cfg.clarify_show]
        key = max(ambiguous, key=len)
        return Clarify(key, shown, keys, more=len(tied) > cfg.clarify_show)

    # --- 問い ---

    def label(self, r: ProcedureRecord) -> str:
        return self._rules.labels.get(r.id) or _PAREN_RE.sub("", r.name).strip()

    def _choices(self, c: Clarify) -> list[str]:
        out = [self.label(r) for r in c.candidates]
        return [*out, self._rules.other_choice] if c.more else out

    def question(self, c: Clarify) -> str:
        """問いの文。字幕には括弧書きで番号付きの選択肢を足す（括弧書きは読み上げない）。"""
        choices = self._choices(c)
        spoken = self._rules.question.format(key=c.key, choices="、".join(f"{x}か" for x in choices))
        return spoken + self._numbered(choices)

    def reask(self, c: Clarify) -> str:
        choices = self._choices(c)
        return self._rules.reask.format(choices="".join(f"{x}か" for x in choices)) + self._numbered(choices)

    @staticmethod
    def _numbered(choices: list[str]) -> str:
        return "（" + "　".join(f"{i + 1}. {_PAREN_RE.sub('', x)}" for i, x in enumerate(choices)) + "）"

    # --- 返事 ---

    def pick(self, c: Clarify, reply: str) -> ProcedureRecord | str | None:
        """返事で決まった候補（「ほか」なら OTHER、決まらなければ None）。"""
        t = reply.strip()
        n = len(c.candidates) + (1 if c.more else 0)
        for idx, words in self._rules.ordinals.items():
            if any(w in t for w in words):
                i = idx if idx >= 0 else n + idx
                if 0 <= i < len(c.candidates):
                    return c.candidates[i]
                if c.more and i == len(c.candidates):
                    return OTHER
        if any(w in t for w in self._rules.other_words):
            return OTHER
        scores = [self.score(r, t, c.keys) for r in c.candidates]
        top = max(scores)
        if top > 0 and scores.count(top) == 1:
            return c.candidates[scores.index(top)]
        return None
