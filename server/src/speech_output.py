"""応答音声の出力先（1 応答 = 複数の文を順に流す）。

- ``LocalSpeaker``: この機体のスピーカーで切れ目なく再生する（StreamPlayer）
- ``WavDirWriter``: 再生せず、1 応答ぶんを 1 つの WAV に繋いで書き出す（無人での自動確認用）
- ``EdgeSpeaker``: エッジ（ブラウザ）へ ``speak``（口パク付き）+ WAV を送り、エッジで鳴らす

いずれも ``push()`` で 1 文ずつ受け取り、``finish()`` で「鳴らし終わった」まで待つ。
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from .edge_bridge import EdgeBridge
from .lipsync import mouth_timeline
from .tts_client import StreamPlayer, TTSClient

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


class EdgeSpeaker:
    """エッジで再生する。``finish()`` はエッジの ``playback_ended`` を待つ。

    エッジは届いた順に文を隙間なく鳴らす前提で、再生終了の見込みを積み上げて返す。
    ``playback_ended`` が来ない（エッジの切断・タブの停止）場合も、
    見込み + timeout_margin_sec で待つのをやめる。
    """

    def __init__(self, bridge: EdgeBridge, timeout_margin_sec: float) -> None:
        self._bridge = bridge
        self._margin = timeout_margin_sec
        self._end_at = 0.0
        self._utterance_id: str | None = None
        self._last_seq = -1
        self._spoken: list[str] = []

    async def push(self, sentence: SpokenSentence) -> float:
        duration = TTSClient.wav_duration_sec(sentence.wav)
        header = {
            "type": "speak",
            "utterance_id": sentence.utterance_id,
            "seq": sentence.seq,
            "text": sentence.text,
            "mouth": mouth_timeline(sentence.query),
        }
        if self._utterance_id is None:
            await self._bridge.send_state("speaking")
        await self._bridge.send_speak(header, sentence.wav)
        self._spoken.append(sentence.text)
        await self._bridge.send_caption("".join(self._spoken), final=False, role="assistant")
        self._utterance_id = sentence.utterance_id
        self._last_seq = sentence.seq
        now = time.monotonic()
        self._end_at = max(now, self._end_at) + duration
        return self._end_at - now

    async def finish(self) -> None:
        if self._utterance_id is None:
            return
        await self._bridge.send_caption("".join(self._spoken), final=True, role="assistant")
        timeout = max(0.0, self._end_at - time.monotonic()) + self._margin
        ok = await self._bridge.wait_playback_ended(self._utterance_id, self._last_seq, timeout)
        if not ok:
            logger.warning(
                "エッジから playback_ended が %.1f秒来ません（%s seq=%d）。再生済みとみなします",
                timeout, self._utterance_id, self._last_seq,
            )
        await self._bridge.send_state("idle")
