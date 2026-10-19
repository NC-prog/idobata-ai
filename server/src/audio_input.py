"""音声入力ソース。

マイクまたは WAV ファイルから 16kHz モノラル float32 のフレームを
AsyncIterator として供給する。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import AudioInputConfig

logger = logging.getLogger(__name__)


class AudioSource(Protocol):
    """音声入力ソースのインターフェース。"""

    def __aiter__(self) -> AsyncIterator[np.ndarray]:
        ...

    async def __anext__(self) -> np.ndarray:
        ...

    async def stop(self) -> None:
        """入力を停止する。"""
        ...


class MicrophoneInput:
    """sounddevice コールバックを使ったマイク入力。

    コールバックスレッドから ``call_soon_threadsafe`` でフレームを
    asyncio.Queue に送る。Queue が満杯の場合はフレームを破棄し警告する。
    """

    def __init__(self, cfg: AudioInputConfig) -> None:
        self._cfg = cfg
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=200)
        self._stream: Any = None  # sd.InputStream
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped_frames = 0
        self._native_rate: int = cfg.sample_rate
        self._resample: bool = False

    async def start(self) -> None:
        """マイク入力ストリームを開始する。

        16kHz で直接オープンを試み、デバイスが対応していなければ
        ネイティブレートで録音しコールバック内でリサンプリングする。
        """
        import sounddevice as sd  # type: ignore[import-untyped]

        self._loop = asyncio.get_running_loop()
        target_rate = self._cfg.sample_rate

        # デバイスのネイティブレートを取得
        dev_info = sd.query_devices(self._cfg.device_id, "input")
        native_rate = int(dev_info["default_samplerate"])
        logger.info(
            "デバイス %s: ネイティブレート %d Hz, 要求 %d Hz",
            dev_info["name"],
            native_rate,
            target_rate,
        )

        # 16kHz で開けるか試す
        try:
            self._open_stream(sd, target_rate)
            self._native_rate = target_rate
            self._resample = False
            logger.info("16kHz で直接オープンしました")
        except (sd.PortAudioError, ValueError):
            # ネイティブレートで開いてリサンプリング
            logger.info(
                "16kHz 非対応。ネイティブレート %d Hz で録音しリサンプリングします",
                native_rate,
            )
            self._native_rate = native_rate
            self._resample = True
            self._open_stream(sd, native_rate)

        self._stream.start()
        logger.info(
            "マイク入力開始 (device=%s, rate=%d→%d, blocksize=%d, resample=%s)",
            self._cfg.device_id,
            self._native_rate,
            target_rate,
            self._cfg.frame_size,
            self._resample,
        )

    def _open_stream(self, sd: Any, samplerate: int) -> None:
        """指定レートで InputStream を生成する。"""
        target_rate = self._cfg.sample_rate
        frame_size = self._cfg.frame_size

        if samplerate != target_rate:
            # ネイティブレートでのブロックサイズを比率で算出
            native_blocksize = int(frame_size * samplerate / target_rate)
        else:
            native_blocksize = frame_size

        resample = samplerate != target_rate

        def _enqueue(mono: np.ndarray) -> None:
            """イベントループスレッドで Queue に投入する。"""
            try:
                self._queue.put_nowait(mono)
            except asyncio.QueueFull:
                self._dropped_frames += 1
                if self._dropped_frames % 100 == 1:
                    logger.warning(
                        "音声フレームをドロップしました（累計 %d フレーム）",
                        self._dropped_frames,
                    )

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
            # モノラル化
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten()

            if resample:
                # 線形補間でリサンプリング
                mono = self._resample_frame(mono, samplerate, target_rate, frame_size)

            assert self._loop is not None
            # コールバックスレッドからイベントループスレッドへ安全に送る
            self._loop.call_soon_threadsafe(_enqueue, mono)

        self._stream = sd.InputStream(
            samplerate=samplerate,
            blocksize=native_blocksize,
            channels=self._cfg.channels,
            dtype="float32",
            device=self._cfg.device_id,
            callback=_callback,
        )

    @staticmethod
    def _resample_frame(
        frame: np.ndarray,
        src_rate: int,
        dst_rate: int,
        dst_size: int,
    ) -> np.ndarray:
        """フレームを線形補間でリサンプリングする。"""
        indices = np.linspace(0, len(frame) - 1, dst_size).astype(np.float64)
        idx_floor = indices.astype(np.int64)
        idx_ceil = np.minimum(idx_floor + 1, len(frame) - 1)
        frac = (indices - idx_floor).astype(np.float32)
        return frame[idx_floor] * (1 - frac) + frame[idx_ceil] * frac

    def __aiter__(self) -> AsyncIterator[np.ndarray]:
        return self

    async def __anext__(self) -> np.ndarray:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def stop(self) -> None:
        """マイク入力を停止する。"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # 終端マーカーを投入して __anext__ を停止させる
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        logger.info("マイク入力停止 (ドロップ: %d フレーム)", self._dropped_frames)


class WavFileInput:
    """WAV ファイルからフレームを供給する。

    16kHz モノラル float32 にリサンプリングし、frame_size サンプルごとに
    asyncio.sleep でペース制御しながら yield する。
    """

    def __init__(self, path: Path, cfg: AudioInputConfig) -> None:
        self._path = path
        self._cfg = cfg
        self._frames: list[np.ndarray] = []
        self._index = 0
        self._stopped = False

    async def start(self) -> None:
        """WAV ファイルを読み込みフレームに分割する。"""
        import soundfile as sf  # type: ignore[import-untyped]

        data, sr = sf.read(str(self._path), dtype="float32")

        # ステレオ → モノラル
        if data.ndim > 1:
            data = data.mean(axis=1)

        # リサンプリング（必要な場合）
        if sr != self._cfg.sample_rate:
            ratio = self._cfg.sample_rate / sr
            new_len = int(len(data) * ratio)
            indices = np.linspace(0, len(data) - 1, new_len).astype(np.int64)
            data = data[indices]
            logger.info("WAV リサンプリング: %d Hz → %d Hz", sr, self._cfg.sample_rate)

        # frame_size ごとに分割
        frame_size = self._cfg.frame_size
        self._frames = []
        for i in range(0, len(data), frame_size):
            chunk = data[i : i + frame_size]
            if len(chunk) < frame_size:
                # 末尾をゼロパディング
                padded = np.zeros(frame_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            self._frames.append(chunk)

        self._index = 0
        logger.info(
            "WAV 読み込み完了: %s (%d フレーム, %.1f秒)",
            self._path.name,
            len(self._frames),
            len(self._frames) * frame_size / self._cfg.sample_rate,
        )

    def __aiter__(self) -> AsyncIterator[np.ndarray]:
        return self

    async def __anext__(self) -> np.ndarray:
        if self._stopped or self._index >= len(self._frames):
            raise StopAsyncIteration

        frame = self._frames[self._index]
        self._index += 1

        # ペース制御（実時間の wav_speed 倍速）
        frame_duration = self._cfg.frame_size / self._cfg.sample_rate
        await asyncio.sleep(frame_duration / self._cfg.wav_speed)

        return frame

    async def stop(self) -> None:
        """入力を停止する。"""
        self._stopped = True
        logger.info("WAV 入力停止")


def list_audio_devices() -> None:
    """利用可能なオーディオデバイスを詳細表示する。"""
    import sounddevice as sd

    devices = sd.query_devices()
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]

    print("=" * 70)
    print("  入力デバイス（--audio-device で指定）")
    print("=" * 70)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *** DEFAULT ***" if i == default_input else ""
            print(
                f"  [{i}] {dev['name']}{marker}\n"
                f"       入力ch: {dev['max_input_channels']}, "
                f"デフォルトレート: {dev['default_samplerate']:.0f} Hz"
            )
    print()
    print("=" * 70)
    print("  出力デバイス（--output-device で指定）")
    print("=" * 70)
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            marker = " *** DEFAULT ***" if i == default_output else ""
            print(
                f"  [{i}] {dev['name']}{marker}\n"
                f"       出力ch: {dev['max_output_channels']}, "
                f"デフォルトレート: {dev['default_samplerate']:.0f} Hz"
            )
    print()
    print("=" * 70)
    print(f"  デフォルト入力: [{default_input}]  デフォルト出力: [{default_output}]")
    print("  使用例: --audio-device 0 --output-device 5")
    print("=" * 70)
