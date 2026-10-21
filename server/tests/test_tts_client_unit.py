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


# ------------------------------------------------------------------
# StreamPlayer（WP01: 文ごとの WAV を 1 本のストリームで連続再生）
# ------------------------------------------------------------------


class TestStreamPlayer:
    """複数の WAV を 1 本の OutputStream で鳴らし切ること。"""

    async def test_plays_all_sentences_on_one_stream(self, fake_stream: Any) -> None:
        client = TTSClient(output_device_id=99)
        player = client.open_player()
        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}):
            r1 = player.push(_make_wav_bytes(samples=2400, samplerate=24000))  # 0.1 秒
            await asyncio.sleep(0.05)  # 2 文目の合成待ち（無音で繋ぐ）
            r2 = player.push(_make_wav_bytes(samples=4800, samplerate=24000))  # 0.2 秒
            await player.finish()

        (stream,) = fake_stream.instances  # 開き直さない
        assert stream.device == 99
        assert stream.channels == 2
        # 0.1 秒 + 0.2 秒 = 7200 サンプル（24kHz のまま）、ほかに合成待ちの無音
        assert stream.written >= 7200 + 480
        assert stream.calls == ["start", "stop", "close"]
        assert 0.09 <= r1 <= 0.11
        assert r2 > 0.2
        await client.close()

    async def test_falls_back_to_default_device(self, fake_stream: Any) -> None:
        fake_stream.fail_devices = {99}
        client = TTSClient(output_device_id=99)
        player = client.open_player()
        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module()}):
            player.push(_make_wav_bytes(samples=2400, samplerate=24000))
            await player.finish()
        (stream,) = fake_stream.instances
        assert stream.device is None
        assert stream.written == 2400  # 既定デバイスでは変換しない
        await client.close()

    async def test_finish_without_push_is_noop(self) -> None:
        client = TTSClient()
        await client.open_player().finish()
        await client.close()

    async def test_hanging_stream_is_aborted(self, fake_stream: Any) -> None:
        client = TTSClient(output_device_id=99)
        client._play_timeout_margin = 0.2
        player = client.open_player()
        with patch.dict("sys.modules", {"sounddevice": _fake_sd_module(_HangingOutputStream)}):
            player.push(_make_wav_bytes(samples=2400, samplerate=24000))
            t0 = time.monotonic()
            await player.finish()
            elapsed = time.monotonic() - t0
            for _ in range(50):
                if "close" in fake_stream.instances[0].calls:
                    break
                await asyncio.sleep(0.01)
        assert elapsed < 1.0
        assert fake_stream.instances[0].calls[-2:] == ["abort", "close"]
        await client.close()


# ------------------------------------------------------------------
# 合成リクエスト・ユーザー辞書（WP01）
# ------------------------------------------------------------------


def _mock_voicevox(
    handler: Any,
) -> TTSClient:
    import httpx

    client = TTSClient()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


class TestSynthesize:
    async def test_sets_post_phoneme_length_and_returns_query(self) -> None:
        import json

        import httpx

        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/audio_query":
                return httpx.Response(200, json={"accent_phrases": [], "postPhonemeLength": 0.5})
            seen["query"] = json.loads(req.content)
            return httpx.Response(200, content=b"RIFF")

        client = _mock_voicevox(handler)
        wav, query = await client.synthesize_with_query("こんにちは")
        assert wav == b"RIFF"
        assert seen["query"]["postPhonemeLength"] == 0.1
        assert query["postPhonemeLength"] == 0.1
        await client.close()

    async def test_retries_server_error(self) -> None:
        import httpx

        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req.url.path)
            if len(calls) == 1:
                return httpx.Response(503)
            if req.url.path == "/audio_query":
                return httpx.Response(200, json={})
            return httpx.Response(200, content=b"RIFF")

        client = _mock_voicevox(handler)
        assert await client.synthesize("あ") == b"RIFF"
        assert calls == ["/audio_query", "/audio_query", "/synthesis"]
        await client.close()

    async def test_client_error_is_not_retried(self) -> None:
        import httpx

        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req.url.path)
            return httpx.Response(422)

        client = _mock_voicevox(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client.synthesize("あ")
        assert calls == ["/audio_query"]
        await client.close()


class TestSyncUserDict:
    async def test_adds_new_updates_changed_and_skips_same(self) -> None:
        import httpx

        from src.site_context import SiteWord

        existing = {
            "u-same": {"surface": "江田島", "pronunciation": "エタジマ", "accent_type": 0,
                       "priority": 9},
            "u-old": {"surface": "能美", "pronunciation": "ノミ", "accent_type": 1,
                      "priority": 5},
        }
        calls: list[tuple[str, str, dict[str, str]]] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append((req.method, req.url.path, dict(req.url.params)))
            if req.method == "GET":
                return httpx.Response(200, json=existing)
            return httpx.Response(200, json="new-uuid")

        client = _mock_voicevox(handler)
        words = [
            SiteWord("江田島", "エタジマ", 0),
            SiteWord("能美", "ノウミ", 0),
            SiteWord("小用", "コヨウ"),  # accent 省略 → 既定値
            SiteWord("市民課"),  # 読みなし → 登録しない
        ]
        assert await client.sync_user_dict(words) == 2
        assert calls[0][:2] == ("GET", "/user_dict")
        put = [c for c in calls if c[0] == "PUT"]
        post = [c for c in calls if c[0] == "POST"]
        assert put == [("PUT", "/user_dict_word/u-old", {
            "surface": "能美", "pronunciation": "ノウミ", "accent_type": "0",
            "word_type": "PROPER_NOUN", "priority": "9",
        })]
        assert len(post) == 1
        assert post[0][2]["surface"] == "小用"
        assert post[0][2]["accent_type"] == "1"
        await client.close()

    async def test_unreachable_voicevox_returns_zero(self) -> None:
        import httpx

        from src.site_context import SiteWord

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = _mock_voicevox(handler)
        assert await client.sync_user_dict([SiteWord("能美", "ノウミ")]) == 0
        await client.close()
