"""tts_client の単体テスト — モノ→ステレオ変換・フォールバック。"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tts_client import TTSClient

# ------------------------------------------------------------------
# ヘルパー: 最小限の WAV バイナリ生成
# ------------------------------------------------------------------

def _make_wav_bytes(samples: int = 480, channels: int = 1, samplerate: int = 24000) -> bytes:
    """soundfile で読めるモノラル / ステレオの WAV バイナリを生成する。"""
    import soundfile as sf  # type: ignore[import-untyped]

    if channels == 1:
        data = np.linspace(-0.5, 0.5, samples, dtype=np.float32)
    else:
        mono = np.linspace(-0.5, 0.5, samples, dtype=np.float32)
        data = np.column_stack([mono] * channels)
    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


# ------------------------------------------------------------------
# _prepare_audio のテスト
# ------------------------------------------------------------------


class _FakeSd:
    """sounddevice のモック。必要最小限のメソッドのみ再現する。"""

    PortAudioError = type("PortAudioError", (Exception,), {})

    def __init__(
        self,
        max_output_channels: int = 2,
        default_samplerate: float = 48000.0,
        check_raises: bool = False,
    ) -> None:
        self._max_output_channels = max_output_channels
        self._default_samplerate = default_samplerate
        self._check_raises = check_raises

    def query_devices(self, device: int, kind: str) -> dict[str, Any]:
        return {
            "name": f"FakeDevice-{device}",
            "max_output_channels": self._max_output_channels,
            "default_samplerate": self._default_samplerate,
        }

    def check_output_settings(self, *, device: int, samplerate: float) -> None:
        if self._check_raises:
            raise self.PortAudioError("unsupported samplerate")


class TestPrepareAudio:
    """_prepare_audio() のチャネル変換・リサンプリングの正しさを検証。"""

    def _make_mono_data(self, n: int = 100) -> np.ndarray:
        return np.linspace(-1.0, 1.0, n, dtype=np.float32)

    def test_mono_to_stereo_shape(self) -> None:
        """モノラル → 2ch デバイスで (N, 2) になること。"""
        client = TTSClient(output_device_id=5)
        sd = _FakeSd(max_output_channels=2)
        mono = self._make_mono_data(200)
        out, _sr = client._prepare_audio(sd, mono, 48000, 5)
        assert out.ndim == 2
        assert out.shape == (200, 2)

    def test_mono_to_stereo_data_identity(self) -> None:
        """ステレオ化後の L/R が元データと完全一致すること。"""
        client = TTSClient(output_device_id=5)
        sd = _FakeSd(max_output_channels=2)
        mono = self._make_mono_data(150)
        out, _ = client._prepare_audio(sd, mono, 48000, 5)
        np.testing.assert_array_equal(out[:, 0], mono)
        np.testing.assert_array_equal(out[:, 1], mono)

    def test_no_conversion_for_1ch_device(self) -> None:
        """1ch デバイスではチャネル変換しないこと。"""
        client = TTSClient(output_device_id=3)
        sd = _FakeSd(max_output_channels=1)
        mono = self._make_mono_data(100)
        out, _ = client._prepare_audio(sd, mono, 48000, 3)
        assert out.ndim == 1
        assert out.shape == (100,)

    def test_already_stereo_no_change(self) -> None:
        """入力が既にステレオなら変換しないこと。"""
        client = TTSClient(output_device_id=5)
        sd = _FakeSd(max_output_channels=2)
        stereo = np.column_stack([
            self._make_mono_data(80),
            self._make_mono_data(80) * 0.5,
        ])
        out, _ = client._prepare_audio(sd, stereo, 48000, 5)
        assert out.shape == (80, 2)
        np.testing.assert_array_equal(out, stereo)

    def test_default_device_skips_ch_conversion(self) -> None:
        """device_id=None（デフォルトデバイス）ではチャネル変換しない。"""
        client = TTSClient(output_device_id=None)
        sd = _FakeSd(max_output_channels=2)
        mono = self._make_mono_data(100)
        out, _ = client._prepare_audio(sd, mono, 48000, None)
        assert out.ndim == 1

    def test_resample_then_stereo(self) -> None:
        """リサンプリングとステレオ化が組み合わせて動作すること。"""
        client = TTSClient(output_device_id=7)
        sd = _FakeSd(max_output_channels=2, default_samplerate=48000.0, check_raises=True)
        mono = self._make_mono_data(240)  # 24000 Hz 想定
        out, sr = client._prepare_audio(sd, mono, 24000, 7)
        # レート変換: 240 samples @ 24kHz → 480 samples @ 48kHz
        assert sr == 48000
        assert out.shape[0] == 480
        # ステレオ化
        assert out.ndim == 2
        assert out.shape[1] == 2

    def test_info_log_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        """変換内容が INFO ログに出力されること。"""
        import logging

        client = TTSClient(output_device_id=5)
        sd = _FakeSd(max_output_channels=2, default_samplerate=48000.0, check_raises=True)
        mono = self._make_mono_data(240)
        with caplog.at_level(logging.INFO, logger="src.tts_client"):
            client._prepare_audio(sd, mono, 24000, 5)
        assert any("ch=1→2" in r.message for r in caplog.records)
        assert any("rate=24000→48000" in r.message for r in caplog.records)


# ------------------------------------------------------------------
# _play_wav フォールバックのテスト
# ------------------------------------------------------------------


class _FakeOutputStream:
    """sounddevice.OutputStream のモック。生成時の device と write 量を記録する。"""

    instances: ClassVar[list[_FakeOutputStream]] = []
    fail_devices: ClassVar[set[int | None]] = set()

    def __init__(
        self, *, samplerate: int, channels: int, dtype: str, device: int | None,
    ) -> None:
        if device in self.fail_devices:
            raise RuntimeError("device unavailable")
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.written = 0
        self.calls: list[str] = []
        _FakeOutputStream.instances.append(self)

    def start(self) -> None:
        self.calls.append("start")

    def write(self, data: np.ndarray) -> None:
        assert data.shape[1] == self.channels
        self.written += data.shape[0]

    def stop(self) -> None:
        self.calls.append("stop")

    def abort(self) -> None:
        self.calls.append("abort")

    def close(self) -> None:
        self.calls.append("close")


def _fake_sd_module(stream_cls: type[Any] = _FakeOutputStream) -> MagicMock:
    def fake_query(dev: int, kind: str) -> dict[str, Any]:
        return {
            "name": f"FakeDevice-{dev}",
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        }

    def fake_check(*, device: int, samplerate: float) -> None:
        pass

    return MagicMock(
        OutputStream=stream_cls,
        query_devices=fake_query,
        check_output_settings=fake_check,
        PortAudioError=Exception,
    )


@pytest.fixture(autouse=False)
def fake_stream() -> Any:
    _FakeOutputStream.instances = []
    _FakeOutputStream.fail_devices = set()
    yield _FakeOutputStream
    _FakeOutputStream.instances = []
    _FakeOutputStream.fail_devices = set()


class TestPlayWavStream:
    """OutputStream を明示的に開いて流し切ること。"""

    def test_writes_all_frames_then_stops_and_closes(self, fake_stream: Any) -> None:
        client = TTSClient(output_device_id=99)
        wav_data = _make_wav_bytes(samples=24000, channels=1, samplerate=24000)

        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}):
            client._play_wav(wav_data)

        (stream,) = fake_stream.instances
        assert stream.device == 99
        assert stream.channels == 2  # モノラル → ステレオ
        assert stream.written == 24000
        assert stream.calls == ["start", "stop", "close"]

    def test_logs_start_and_end(
        self, fake_stream: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        client = TTSClient(output_device_id=99)
        wav_data = _make_wav_bytes(samples=2400, channels=1, samplerate=24000)
        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}), \
                caplog.at_level(logging.INFO, logger="src.tts_client"):
            client._play_wav(wav_data)

        messages = [r.message for r in caplog.records]
        assert any(m.startswith("再生開始: device=99") for m in messages)
        assert any(m.startswith("再生終了: device=99, 経過") for m in messages)


class TestPlayWavFallback:
    """指定デバイスで再生失敗時のデフォルトデバイスフォールバックを検証。"""

    def test_fallback_to_default_device(self, fake_stream: Any) -> None:
        """指定デバイスで例外発生時、device=None で再試行すること。"""
        client = TTSClient(output_device_id=99)
        wav_data = _make_wav_bytes(samples=480, channels=1, samplerate=24000)
        fake_stream.fail_devices = {99}

        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}):
            client._play_wav(wav_data)

        (stream,) = fake_stream.instances
        assert stream.device is None
        assert stream.calls == ["start", "stop", "close"]

    def test_fallback_emits_warning(
        self, fake_stream: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """フォールバック時に WARNING ログが出ること。"""
        import logging

        client = TTSClient(output_device_id=99)
        wav_data = _make_wav_bytes(samples=480, channels=1, samplerate=24000)
        fake_stream.fail_devices = {99}

        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}), \
                caplog.at_level(logging.WARNING, logger="src.tts_client"):
            client._play_wav(wav_data)

        assert any("フォールバック" in r.message for r in caplog.records)


class _HangingOutputStream(_FakeOutputStream):
    """write() が abort() されるまで戻らないストリーム（実機の不具合の再現）。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._released = threading.Event()

    def write(self, data: np.ndarray) -> None:
        self._released.wait(timeout=10)
        if "abort" in self.calls:
            raise RuntimeError("stream aborted")

    def abort(self) -> None:
        super().abort()
        self._released.set()


class TestPlayTimeout:
    """play() が戻らない再生を打ち切ること。"""

    async def test_hanging_stream_is_aborted_after_timeout(
        self, fake_stream: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        client = TTSClient(output_device_id=99)
        client._play_timeout_margin = 0.2
        wav_data = _make_wav_bytes(samples=2400, channels=1, samplerate=24000)  # 0.1 秒

        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module(_HangingOutputStream)}), \
                caplog.at_level(logging.WARNING, logger="src.tts_client"):
            t0 = time.monotonic()
            await client.play(wav_data)
            elapsed = time.monotonic() - t0
            # 強制クローズはデーモンスレッドで行うので、少し待ってから確認する
            for _ in range(50):
                if "close" in fake_stream.instances[0].calls:
                    break
                await asyncio.sleep(0.01)

        assert 0.25 <= elapsed < 1.0  # WAV 0.1 秒 + 0.2 秒で打ち切り
        (stream,) = fake_stream.instances  # 中断時はフォールバックしない
        assert stream.calls[-2:] == ["abort", "close"]
        assert any("強制的に閉じます" in r.message for r in caplog.records)
        await client.close()

    async def test_normal_play_returns_without_warning(
        self, fake_stream: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        client = TTSClient(output_device_id=99)
        wav_data = _make_wav_bytes(samples=2400, channels=1, samplerate=24000)
        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}), \
                caplog.at_level(logging.WARNING, logger="src.tts_client"):
            await client.play(wav_data)

        assert fake_stream.instances[0].calls == ["start", "stop", "close"]
        assert not caplog.records
        await client.close()

    def test_wav_duration_sec(self) -> None:
        wav_data = _make_wav_bytes(samples=36000, channels=1, samplerate=24000)
        assert TTSClient.wav_duration_sec(wav_data) == pytest.approx(1.5)


# ------------------------------------------------------------------
# _resample_audio の既存テスト（回帰防止）
# ------------------------------------------------------------------


class TestResampleAudio:
    """_resample_audio() の基本動作を検証。"""

    def test_upsample_mono(self) -> None:
        mono = np.arange(100, dtype=np.float32)
        out = TTSClient._resample_audio(mono, 24000, 48000)
        assert out.shape[0] == 200

    def test_upsample_stereo(self) -> None:
        stereo = np.column_stack([
            np.arange(100, dtype=np.float32),
            np.arange(100, dtype=np.float32) * 2,
        ])
        out = TTSClient._resample_audio(stereo, 24000, 48000)
        assert out.shape == (200, 2)

    def test_identity_when_same_rate(self) -> None:
        mono = np.arange(50, dtype=np.float32)
        out = TTSClient._resample_audio(mono, 48000, 48000)
        assert out.shape[0] == 50
        np.testing.assert_allclose(out, mono, atol=1e-5)
