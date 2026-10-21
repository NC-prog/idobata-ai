"""応答音声の出力先（1 応答 = 複数の文を順に流す）。

- ``LocalSpeaker``: この機体のスピーカーで切れ目なく再生する（StreamPlayer）
- ``WavDirWriter``: 再生せず、1 応答ぶんを 1 つの WAV に繋いで書き出す（無人での自動確認用）

どちらも ``push()`` で 1 文ずつ受け取り、``finish()`` で「鳴らし終わった」まで待つ。
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from .tts_client import StreamPlayer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpokenSentence:
    """合成済みの 1 文。"""

    utterance_id: str
    seq: int
    text: str
    wav: bytes
    # 合成に使った VOICEVOX の audio_query（口パクの時系列を作る）
    query: dict[str, Any] = field(default_factory=dict)


class SpeechSink(Protocol):
    """応答音声の出力先。"""

    async def push(self, sentence: SpokenSentence) -> float:
        """1 文を流す（1 文目で再生開始）。今から再生終了までの見込み秒数を返す。"""
        ...

    async def finish(self) -> None:
        """流した文をすべて鳴らし終えるまで待つ。"""
        ...


class LocalSpeaker:
    """この機体のスピーカーで再生する。"""

    def __init__(self, player: StreamPlayer) -> None:
        self._player = player

    async def push(self, sentence: SpokenSentence) -> float:
        return self._player.push(sentence.wav)

    async def finish(self) -> None:
        await self._player.finish()


class WavDirWriter:
    """1 応答ぶんの文を 1 つの WAV に繋いで書き出す（評価・自動確認用。会話音声の保存ではない）。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._chunks: list[np.ndarray] = []
        self._rate: int | None = None

    async def push(self, sentence: SpokenSentence) -> float:
        data, rate = sf.read(io.BytesIO(sentence.wav), dtype="float32")
        if self._rate is None:
            self._rate = int(rate)
        elif int(rate) != self._rate:
            raise ValueError(f"文ごとにサンプルレートが違います: {rate} != {self._rate}")
        self._chunks.append(data)
        return 0.0

    async def finish(self) -> None:
        if self._rate is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(self._path), np.concatenate(self._chunks), self._rate)
        print(f"[音声] {self._path}")
