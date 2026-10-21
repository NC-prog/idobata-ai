"""VOICEVOX の audio_query から口パクの時系列を作る（WP01）。

各モーラの子音長・母音長を先頭から積み上げて、母音が鳴っている区間に VRM の口形
（aa / ih / ou / ee / oh）を割り当てる。撥音（N）・促音（cl）・句読点の無音（pause_mora）は
閉口なので出力しない（エッジは区間外を閉口として扱う）。無声化した母音（VOICEVOX では
大文字。「です」の「ス」など）も口をほとんど開けないので出さない。
音量包絡ではなく音素に同期するので、口の開き始めと閉じが音声と合う。
"""

from __future__ import annotations

from typing import Any, TypedDict

# VOICEVOX の母音 → VRM 1.0 の口形プリセット（小文字 = 有声のみ）
VOWEL_TO_VISEME: dict[str, str] = {
    "a": "aa",
    "i": "ih",
    "u": "ou",
    "e": "ee",
    "o": "oh",
}


class MouthKey(TypedDict):
    """口形 1 区間。t は WAV 先頭からの秒、d は長さ（秒）。"""

    t: float
    v: str
    d: float


def mouth_timeline(query: dict[str, Any]) -> list[MouthKey]:
    """audio_query（合成に使ったもの）から口形の区間列を作る。"""
    speed = float(query.get("speedScale") or 1.0)
    t = float(query.get("prePhonemeLength") or 0.0)
    keys: list[MouthKey] = []
    for phrase in query.get("accent_phrases") or []:
        for mora in phrase.get("moras") or []:
            t += float(mora.get("consonant_length") or 0.0)
            vowel_length = float(mora.get("vowel_length") or 0.0)
            viseme = VOWEL_TO_VISEME.get(str(mora.get("vowel", "")))
            if viseme is not None and vowel_length > 0:
                keys.append({"t": round(t / speed, 3), "v": viseme,
                             "d": round(vowel_length / speed, 3)})
            t += vowel_length
        pause = phrase.get("pause_mora")
        if pause:
            t += float(pause.get("consonant_length") or 0.0) + float(pause.get("vowel_length") or 0.0)
    return keys
