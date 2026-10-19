"""speech_gate.py のテスト。"""

from __future__ import annotations

import asyncio
import logging

import pytest

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


async def test_safety_unmute_when_play_never_returns() -> None:
    gate = SpeechGate(post_margin_sec=0.02, safety_margin_sec=0.02)
    gate.mute()
    gate.mute(expected_sec=0.02)  # 合成後に想定再生時間を渡す。schedule_unmute() は来ない
    await asyncio.sleep(0.03)
    assert gate.is_muted
    await asyncio.sleep(0.06)
    assert not gate.is_muted


async def test_schedule_unmute_takes_priority_over_safety() -> None:
    gate = SpeechGate(post_margin_sec=0.05, safety_margin_sec=0.0)
    gate.mute(expected_sec=0.0)  # 安全装置は 0.05 秒後
    gate.schedule_unmute()  # 通常の unmute（0.05 秒後）。安全装置は取り消される
    await asyncio.sleep(0.02)
    gate.mute()  # 次の発話: 通常 unmute も取り消される
    await asyncio.sleep(0.1)
    assert gate.is_muted  # 安全装置が残っていたら開いてしまう


async def test_safety_unmute_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    gate = SpeechGate(post_margin_sec=0.0, safety_margin_sec=0.0)
    with caplog.at_level(logging.WARNING, logger="src.speech_gate"):
        gate.mute(expected_sec=0.01)
        await asyncio.sleep(0.05)
    assert not gate.is_muted
    assert any("安全装置" in r.message for r in caplog.records)
