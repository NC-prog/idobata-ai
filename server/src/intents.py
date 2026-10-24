"""来訪者の発話の型（WP13 B）。LLM を使わず、語で決める（server/context/intents.yaml）。

型ごとに、毎ターンの検索で渡してよい知識の範囲が決まる（KnowledgeConfig.scope_by_intent）。
8B は渡された資料を本題にしてしまうので、型に合わない資料はそもそも渡さない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .procedures import Procedures
from .site_context import Facility

PROCEDURE = "procedure"
FACILITY = "facility"
EVENT = "event"
TOURISM = "tourism"
WEATHER = "weather"
OTHER = "other"
SMALLTALK = "smalltalk"
# 1 発話に複数当たったときの優先（前ほど強い）
PRIORITY = (PROCEDURE, FACILITY, EVENT, TOURISM, WEATHER, OTHER, SMALLTALK)
_WORD_KEYS = (PROCEDURE, FACILITY, EVENT, TOURISM, WEATHER)

# 名詞らしい語（漢字・カタカナの 2 字以上の連なり。漢字 1 字なら送り仮名 1 字まで含める: 祭り）。形態素解析は使わない。
# 資料の足切り（B-2）と、名詞だけの短い発話（「印鑑登録」）を問いとみなすのに使う
_NOUN_RE = re.compile(r"[ァ-ヴー]{2,}|[一-龥々ヶ]{2,}|[一-龥々ヶ](?![かがのはをにでともへやねよ])[ぁ-ゖ]")
# 名詞として数えない語（問い・挨拶の決まり文句の漢字）
_NOT_NOUNS = frozenset({"今日", "明日", "最近", "大丈夫", "失礼", "本当", "何時", "以上", "場合", "時間", "ところ",
                        "何か", "何が", "何を", "何の", "何で", "誰か", "私は", "私の", "分か", "教え", "聞き", "来ま", "行き"})


@dataclass(frozen=True)
class IntentRules:
    words: dict[str, tuple[str, ...]]

    def hits(self, text: str) -> set[str]:
        return {k for k, ws in self.words.items() if any(w in text for w in ws)}


def load_intent_rules(path: Path) -> IntentRules:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - set(_WORD_KEYS)
    if unknown:
        raise ValueError(f"intents.yaml の型は {_WORD_KEYS} のどれか: {sorted(unknown)}")
    return IntentRules({k: tuple(str(w) for w in data.get(k) or []) for k in _WORD_KEYS})


def nouns(text: str) -> list[str]:
    """発話の名詞らしい語（2 字以上の漢字・カタカナ）。"""
    return [w for w in dict.fromkeys(_NOUN_RE.findall(text)) if w not in _NOT_NOUNS]


def noun_heavy(text: str) -> bool:
    """名詞が中心の短い発話か（「印鑑登録」「住民票の写し」。問いとみなす）。"""
    body = re.sub(r"[\s、。！？!?…ー〜]", "", text)
    ns = nouns(text)
    return bool(body) and len(body) <= 12 and sum(len(n) for n in ns) * 2 >= len(body)


def procedure_surfaces(text: str, procedures: Procedures) -> list[str]:
    """発話に出てきた、別名表と手続きナレッジ（承認済み・下書き）の名前・別名。"""
    found = [w for r in procedures.matched(text) for w in r.surfaces if w in text]
    return found + [w for w in procedures.surfaces() if w in text]


def procedure_names(text: str, procedures: Procedures) -> list[str]:
    """発話で当たった手続きの、名前と別名のすべて（「国保」と言えば「国民健康保険」も。資料の足切りの語に使う）。"""
    out = [w for r in procedures.matched(text) for w in (r.name, *r.surfaces)]
    out += [w for p in procedures.items for w in (p.term, *p.aliases)
            if any(s in text for s in (p.term, *p.aliases))]
    return list(dict.fromkeys(w for w in out if w))


def _tourism_word(surface: str, rules: IntentRules) -> bool:
    """観光の語を含む名前・別名か（下書き「観光案内」の別名「観光」「観光地」で、観光の問いを手続きにしない）。"""
    return any(w in surface for w in rules.words[TOURISM])


def classify_intent(text: str, rules: IntentRules, procedures: Procedures | None, facility: Facility | None,
                    question_re: re.Pattern[str]) -> str:
    """発話の型。手続きは語に加えて、別名表・手続きナレッジ（承認済み・下書き）の名前と別名で当たる。"""
    del facility  # 課の担当業務（handles）の語は使わない（下の注記）
    t = text.strip()
    found = rules.hits(t)
    # 課の担当業務（handles）の語は見ない（商工観光課の「観光」で、観光の問いが手続きになる）
    if procedures is not None and any(not _tourism_word(s, rules) for s in procedure_surfaces(t, procedures)):
        found.add(PROCEDURE)
    for intent in PRIORITY[:5]:
        if intent in found:
            return intent
    return OTHER if question_re.search(t) or noun_heavy(t) else SMALLTALK
