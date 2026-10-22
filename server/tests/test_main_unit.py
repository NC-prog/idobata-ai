"""main.VoiceLoop の待機タイムアウト（WP01 追加作業）のテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.main import VoiceLoop
from src.transcripts import TranscriptStore


class _Recorder:
    edge_id = "edge-test"

    def __init__(self) -> None:
        self.states: list[str] = []
        self.resets = 0

    async def send_state(self, value: str) -> None:
        self.states.append(value)

    def reset(self) -> None:
        self.resets += 1

    async def send_caption(self, text: str, final: bool = True, role: str = "assistant") -> None:
        pass


def _loop() -> tuple[VoiceLoop, _Recorder]:
    app = VoiceLoop(mode="voice", input_wavs=[], audio_device=None, output_device=None,
                    output_wav_dir=None)
    rec = _Recorder()
    app._edge = rec  # type: ignore[assignment]
    app._dialogue.reset = rec.reset  # type: ignore[method-assign]
    # 文字記録は作業用のディレクトリに書く（実データの ~/idobata-data に書かない）
    app._transcripts = TranscriptStore(Path(tempfile.mkdtemp(prefix="idobata-test-")), "etajima", 7)
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


# ------------------------------------------------------------------
# 短い返事の拾い上げ（WP05-7）
# ------------------------------------------------------------------


class _FakeASR:
    def __init__(self, outputs: list[tuple[str, str]]) -> None:
        self.outputs = outputs
        self.forced: list[str | None] = []
        self.seconds: list[float] = []

    async def transcribe_segment(self, frames, t, sample_rate=16000, force_language=None):  # type: ignore[no-untyped-def]
        from src.asr_client import TranscriptChunk

        self.forced.append(force_language)
        self.seconds.append(sum(len(f) for f in frames) / sample_rate)
        text, lang = self.outputs.pop(0)
        yield TranscriptChunk(text=text, is_final=True, t_speech_start=t, t_emitted=t, language=lang)

    def push_rolling_context(self, text: str) -> None:
        pass


def _voiced_frames(n_voiced: int) -> list:  # type: ignore[type-arg]
    import numpy as np

    return [np.full(512, 0.2, np.float32) if i < n_voiced else np.zeros(512, np.float32) for i in range(n_voiced + 2)]


async def test_short_foreign_reply_is_retried_in_session_language() -> None:
    app, _ = _loop()
    heard: list[str] = []

    async def fake(text: str, language: str, timing: object) -> None:
        heard.append(text)

    app._respond_and_speak = fake  # type: ignore[method-assign,assignment]
    app._asr = _FakeASR([("嗯。", "zh"), ("うん。", "ja")])  # type: ignore[assignment]
    app._conversation._ensure_session("ja")
    assert app._conversation._session is not None
    app._conversation._session.turns = 1  # 日本語の会話の途中
    await app._on_segment(_voiced_frames(8), 0.0)
    await app._drain_turns()
    assert app._asr.forced == [None, "Japanese"]  # type: ignore[attr-defined]
    assert app._asr.seconds[0] >= 0.8  # 短い区間は無音を足して 0.8 秒以上
    assert heard == ["うん。"]


async def test_reply_phase_lowers_vad_threshold_and_pads_to_one_second() -> None:
    from src.actions import Pending
    from src.config import config

    app, _ = _loop()
    app._respond_and_speak = lambda *a: _noop()  # type: ignore[method-assign,assignment,return-value]
    app._asr = _FakeASR([("うん。", "ja")])  # type: ignore[assignment]
    app._conversation._pending = Pending("lookup", "x", "ja")
    await app._on_segment(_voiced_frames(6), 0.0)
    await app._drain_turns()
    assert app._asr.forced == ["Japanese"]  # type: ignore[attr-defined]
    assert app._asr.seconds[0] >= 1.0  # type: ignore[attr-defined]

    class _VAD:
        in_speech = False

        def __init__(self) -> None:
            self.thresholds: list[float] = []

        def set_threshold(self, t: float) -> None:
            self.thresholds.append(t)

        def process_frame(self, frame: object) -> None:
            return None

        def reset(self) -> None:
            pass

    async def frames():  # type: ignore[no-untyped-def]
        import numpy as np

        yield np.zeros(512, np.float32)

    vad = _VAD()
    await app._run_pipeline(vad, frames())  # type: ignore[arg-type]
    assert vad.thresholds == [config.vad.reply_threshold]
    app._conversation._pending = None
    await app._run_pipeline(vad, frames())  # type: ignore[arg-type]
    assert vad.thresholds[-1] == config.vad.threshold


async def _noop() -> None:
    return None


async def test_phone_reask_waits_until_visitor_stops_speaking() -> None:
    """電話番号の聞き直しは、来訪者の発話区間（と聞き取り）が終わってから言う（WP06）。"""
    import asyncio
    import time

    from src.actions import Pending

    app, _rec = _loop()
    p = Pending("refer", "国保の相談", "ja", phase="ask_phone")
    p.phone_deadline = time.monotonic() - 1  # 期限は過ぎている
    app._conversation._pending = p
    calls: list[bool] = []

    async def fake_respond(text: str, language: str, timing: object, source: object = None) -> None:
        calls.append(app._hearing)
        p.phone_deadline = None  # 聞き直した

    app._respond_and_speak = fake_respond  # type: ignore[method-assign]
    app._hearing = True  # 来訪者が話している途中
    task = asyncio.create_task(app._watch_phone_deadline())
    await asyncio.sleep(0.3)
    assert calls == []  # 話している間は言わない
    app._hearing = False
    await asyncio.wait_for(task, 1.0)
    assert calls == [False]
    await app._tts.close()
