"""main.VoiceLoop の待機タイムアウト（WP01 追加作業）のテスト。"""

from __future__ import annotations

from src.main import VoiceLoop


class _Recorder:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.resets = 0

    async def send_state(self, value: str) -> None:
        self.states.append(value)

    def reset(self) -> None:
        self.resets += 1


def _loop() -> tuple[VoiceLoop, _Recorder]:
    app = VoiceLoop(mode="voice", input_wavs=[], audio_device=None, output_device=None,
                    output_wav_dir=None)
    rec = _Recorder()
    app._edge = rec  # type: ignore[assignment]
    app._dialogue.reset = rec.reset  # type: ignore[method-assign]
    return app, rec


async def test_idle_timeout_resets_history_and_sends_idle() -> None:
    app, rec = _loop()
    app._last_utterance_at = 100.0
    assert await app.check_idle(now=150.0, timeout=90.0) is False
    assert await app.check_idle(now=190.0, timeout=90.0) is True
    assert rec.resets == 1 and rec.states == ["idle"]
    # 一度区切ったら、次の発話までは何もしない
    assert await app.check_idle(now=400.0, timeout=90.0) is False
    assert rec.resets == 1
    await app._tts.close()


async def test_idle_timeout_waits_for_running_turn() -> None:
    app, _rec = _loop()
    app._last_utterance_at = 0.0
    async with app._turn_lock:
        assert await app.check_idle(now=500.0, timeout=90.0) is False
    assert await app.check_idle(now=500.0, timeout=90.0) is True
    await app._tts.close()


async def test_no_timeout_before_first_utterance() -> None:
    app, rec = _loop()
    assert await app.check_idle(now=1e9, timeout=90.0) is False
    assert rec.states == []
    await app._tts.close()
