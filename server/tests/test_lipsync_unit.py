"""lipsync（audio_query → 口形の時系列）のテスト。"""

from __future__ import annotations

from typing import Any

import pytest

from src.lipsync import mouth_timeline


def _mora(text: str, consonant: float | None, vowel: str, length: float) -> dict[str, Any]:
    return {"text": text, "consonant_length": consonant, "vowel": vowel, "vowel_length": length}


_QUERY: dict[str, Any] = {
    "prePhonemeLength": 0.1,
    "speedScale": 1.0,
    "accent_phrases": [
        {
            "moras": [
                _mora("ノ", 0.05, "o", 0.1),
                _mora("ウ", None, "u", 0.08),
                _mora("ミ", 0.06, "i", 0.05),
                _mora("デ", 0.04, "e", 0.08),
                _mora("ス", 0.05, "U", 0.09),  # 無声化
            ],
            "pause_mora": {"consonant_length": None, "vowel": "pau", "vowel_length": 0.4},
        },
        {"moras": [_mora("ハ", 0.1, "a", 0.12), _mora("ン", None, "N", 0.07),
                   _mora("ッ", None, "cl", 0.05), _mora("イ", None, "i", 0.12)]},
    ],
}


def test_vowels_mapped_to_visemes_with_cumulative_times() -> None:
    keys = mouth_timeline(_QUERY)
    assert [k["v"] for k in keys] == ["oh", "ou", "ih", "ee", "aa", "ih"]
    # 先頭の無音 0.1 + 子音 0.05 で「オ」が開く
    assert keys[0] == {"t": 0.15, "v": "oh", "d": 0.1}
    assert keys[1]["t"] == pytest.approx(0.25)
    # 句読点の無音 0.4 を挟んだ後の「ハ」: 0.1 + 0.60（1 句目）+ 0.4 + 子音 0.1
    assert keys[4]["t"] == pytest.approx(1.2)
    # 撥音・促音は閉口なので出さず、その長さだけ時間が進む
    assert keys[5]["t"] == pytest.approx(1.2 + 0.12 + 0.07 + 0.05)


def test_speed_scale_shortens_times() -> None:
    fast = mouth_timeline({**_QUERY, "speedScale": 2.0})
    assert fast[0] == {"t": 0.075, "v": "oh", "d": 0.05}


def test_empty_query() -> None:
    assert mouth_timeline({}) == []
