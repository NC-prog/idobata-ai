"""時間帯の挨拶（おはようございます / こんにちは / こんばんは）。

応答（LLM）・相槌・定型文・アナウンスで同じ判定を使う。境目は ``DialogueConfig.greeting_bands``
（5〜10 時 おはようございます、10〜17 時 こんにちは、17 時以降と 5 時より前 こんばんは）。
"""

from __future__ import annotations

from datetime import datetime

from .config import DialogueConfig, config
from .knowledge import JST


def greeting_for(now: datetime | None = None, cfg: DialogueConfig | None = None) -> str:
    """いまの時間帯の挨拶。"""
    bands = (cfg or config.dialogue).greeting_bands
    hour = (now or datetime.now(JST)).hour
    word = bands[-1][1]  # 最初の帯より前（0〜5 時）は最後の帯（こんばんは）のまま
    for start, w in bands:
        if hour >= start:
            word = w
    return word


def all_greetings(cfg: DialogueConfig | None = None) -> list[str]:
    return [w for _, w in (cfg or config.dialogue).greeting_bands]


def fix_greeting(sentence: str, greeting: str, cfg: DialogueConfig | None = None) -> str:
    """ほかの時間帯の挨拶を、いまの挨拶に置き換える（夜に「おはようございます」と言わない）。"""
    for w in all_greetings(cfg):
        if w != greeting:
            sentence = sentence.replace(w, greeting)
    return sentence
