"""speech_gate.py のテスト。"""

from __future__ import annotations

import asyncio

from src.speech_gate import SpeechGate


async def test_mute_then_unmute_after_margin() -> None:
    gate = SpeechGate(post_margin_sec=0.05)
    assert not gate.is_muted
    gate.mute()
    assert gate.is_muted
    gate.schedule_unmute()
    assert gate.is_muted
    await asyncio.sleep(0.1)
    assert not gate.is_muted


async def test_mute_cancels_pending_unmute() -> None:
    gate = SpeechGate(post_margin_sec=0.05)
    gate.mute()
    gate.schedule_unmute()
    gate.mute()
    await asyncio.sleep(0.1)
    assert gate.is_muted
