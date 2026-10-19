"""Silero VAD ラッパー。

sounddevice から受け取った 512 サンプル (16kHz, 32ms) のフレームを逐次処理し、
発話開始 / 終了イベントを生成する。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import numpy as np
import torch

from .config import VADConfig

logger = logging.getLogger(__name__)


class SpeechEventType(Enum):
    """VAD が検出するイベント種別。"""

    SPEECH_START = auto()
    SPEECH_END = auto()


@dataclass(frozen=True)
class SpeechEvent:
    """VAD から発行されるイベント。"""

    event_type: SpeechEventType
    timestamp_sec: float


class VADBackend(Protocol):
    """VAD バックエンドのインターフェース。"""

    @property
    def in_speech(self) -> bool:
        """現在発話中かどうか。"""
        ...

    def process_frame(self, frame: np.ndarray) -> SpeechEvent | None:
        """フレームを処理し、イベントがあれば返す。"""
        ...

    def reset(self) -> None:
        """内部状態をリセットする。"""
        ...


class SileroVAD:
    """Silero VAD をラップする実装。

    silero_vad パッケージの VADIterator を使用し、max_segment_sec 超過時に
    強制分割（SPEECH_END 発行 + reset）を行う。
    """

    def __init__(self, cfg: VADConfig) -> None:
        self._cfg = cfg
        self._sample_rate = cfg.sample_rate
        self._max_segment_samples = int(cfg.max_segment_sec * cfg.sample_rate)

        # ソフト分割: lookback 区間の開始サンプル位置
        lookback_samples = int(cfg.soft_split_lookback_ms * cfg.sample_rate / 1000)
        self._soft_zone_start_samples = max(
            0, self._max_segment_samples - lookback_samples
        )
        self._soft_split_enabled = cfg.soft_split_enabled
        self._soft_split_rise_threshold = cfg.soft_split_rise_threshold

        # silero_vad の VADIterator を初期化
        from silero_vad import VADIterator, load_silero_vad  # type: ignore[import-untyped]

        self._model = load_silero_vad()
        self._iterator = VADIterator(
            model=self._model,
            threshold=cfg.threshold,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.pre_roll_ms,
            sampling_rate=cfg.sample_rate,
        )

        # 発話中フラグと累積サンプル数
        self._in_speech = False
        self._speech_samples = 0
        self._speech_start_time = 0.0

        # ソフト分割の内部状態
        self._min_energy: float = float("inf")
        self._in_soft_zone = False

    def process_frame(self, frame: np.ndarray) -> SpeechEvent | None:
        """フレームを処理し、イベントがあれば返す。

        Args:
            frame: float32 の音声フレーム（512 サンプル、16kHz モノラル）

        Returns:
            SpeechEvent または None
        """
        # numpy → torch テンソル変換
        tensor = torch.from_numpy(frame).float()

        result = self._iterator(tensor, return_seconds=False)

        now = time.monotonic()

        if result is not None:
            if "start" in result:
                self._in_speech = True
                self._speech_samples = 0
                self._speech_start_time = now
                self._reset_soft_state()
                logger.debug("VAD: speech start")
                return SpeechEvent(SpeechEventType.SPEECH_START, now)
            if "end" in result:
                self._in_speech = False
                self._speech_samples = 0
                self._reset_soft_state()
                logger.debug("VAD: speech end")
                return SpeechEvent(SpeechEventType.SPEECH_END, now)

        # 発話中の場合、累積サンプル数を更新
        if self._in_speech:
            self._speech_samples += len(frame)

            # ソフト分割ゾーン判定
            if (
                self._soft_split_enabled
                and self._speech_samples >= self._soft_zone_start_samples
            ):
                rms = float(np.sqrt(np.mean(frame**2)))

                if not self._in_soft_zone:
                    # ソフト分割ゾーンに入った
                    self._in_soft_zone = True
                    self._min_energy = rms

                if rms < self._min_energy:
                    self._min_energy = rms
                elif rms > self._min_energy + self._soft_split_rise_threshold:
                    # 局所最小を通過 → ソフト分割
                    elapsed = self._speech_samples / self._sample_rate
                    logger.info("VAD: soft split at %.1fs", elapsed)
                    self._in_speech = False
                    self._speech_samples = 0
                    self._reset_soft_state()
                    self._iterator.reset_states()
                    return SpeechEvent(SpeechEventType.SPEECH_END, now)

            # max_segment_sec 超過で強制分割（ハードカットオフ）
            if self._speech_samples >= self._max_segment_samples:
                logger.info(
                    "VAD: max_segment_sec (%.1f秒) を超過。強制分割します。",
                    self._cfg.max_segment_sec,
                )
                self._in_speech = False
                self._speech_samples = 0
                self._reset_soft_state()
                self._iterator.reset_states()
                return SpeechEvent(SpeechEventType.SPEECH_END, now)

        return None

    def _reset_soft_state(self) -> None:
        """ソフト分割の内部状態をリセットする。"""
        self._min_energy = float("inf")
        self._in_soft_zone = False

    def reset(self) -> None:
        """内部状態をリセットする。"""
        self._iterator.reset_states()
        self._in_speech = False
        self._speech_samples = 0
        self._speech_start_time = 0.0
        self._reset_soft_state()
        logger.debug("VAD: reset")

    @property
    def in_speech(self) -> bool:
        """現在発話中かどうか。"""
        return self._in_speech


def create_vad(cfg: VADConfig) -> SileroVAD:
    """VADConfig から VAD バックエンドを生成するファクトリ関数。"""
    if cfg.backend != "silero":
        raise ValueError(f"未対応の VAD バックエンド: {cfg.backend}")
    return SileroVAD(cfg)
