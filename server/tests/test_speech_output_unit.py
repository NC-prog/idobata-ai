"""speech_output（応答音声の出力先）のテスト。"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from src.speech_output import SpokenSentence, WavDirWriter


def _wav(n: int, value: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(n, value, dtype=np.float32), rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


async def test_wav_writer_concatenates_sentences(tmp_path: Path) -> None:
    out = tmp_path / "reply_01.wav"
    w = WavDirWriter(out)
    await w.push(SpokenSentence("u1", 0, "あ。", _wav(100, 0.25)))
    await w.push(SpokenSentence("u1", 1, "い。", _wav(50, -0.5)))
    await w.finish()
    data, rate = sf.read(str(out), dtype="float32")
    assert rate == 24000
    assert len(data) == 150
    assert np.allclose(data[:100], 0.25) and np.allclose(data[100:], -0.5)


async def test_wav_writer_without_sentences_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "reply.wav"
    await WavDirWriter(out).finish()
    assert not out.exists()
