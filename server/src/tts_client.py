"""VOICEVOX TTS クライアント。

受付ボットから移植。synthesize_and_play() によるグレースフルデグレード付き
音声再生機能を追加。
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .config import config

logger = logging.getLogger(__name__)


class _PlaybackSession:
    """1 回の再生の中断制御（再生スレッドとイベントループの間で共有する）。"""

    def __init__(self) -> None:
        self.aborted = threading.Event()
        self._lock = threading.Lock()
        self._stream: Any = None

    def attach(self, stream: Any) -> None:
        with self._lock:
            self._stream = stream

    def detach(self) -> None:
        with self._lock:
            self._stream = None

    def abort(self) -> None:
        """中断を要求し、開いているストリームを別スレッドで強制的に閉じる。

        abort()/close() 自体が戻らない場合でもイベントループを止めないよう、
        閉じる処理はデーモンスレッドに任せる。
        """
        self.aborted.set()
        with self._lock:
            stream = self._stream
        if stream is None:
            return

        def _close() -> None:
            try:
                stream.abort()
            except Exception:
                logger.debug("出力ストリームの abort に失敗", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("出力ストリームの close に失敗", exc_info=True)

        threading.Thread(target=_close, name="tts-play-abort", daemon=True).start()


def _resolve_future(fut: asyncio.Future[None], exc: BaseException | None) -> None:
    """再生スレッドから call_soon_threadsafe で呼ぶ。待ち手がもういなければ何もしない。"""
    if fut.done():
        return
    if exc is None:
        fut.set_result(None)
    else:
        fut.set_exception(exc)


class TTSClient:
    """VOICEVOX 音声合成クライアント。"""

    def __init__(self, output_device_id: int | None = None) -> None:
        self.base_url = config.voicevox.base_url
        self.speaker_id = config.voicevox.speaker_id
        self.output_device_id = output_device_id
        self.client = httpx.AsyncClient(timeout=30.0)
        self._prewarmed = False
        self._play_timeout_margin = config.audio_output.play_timeout_margin_sec
        self._write_block_sec = config.audio_output.write_block_sec

    async def prewarm(self) -> bool:
        """VOICEVOX エンジンをプリウォームする（冪等）。

        短いダミーテキストで audio_query + synthesis を1回実行し、
        初回合成時の遅延を回避する。複数回呼んでも安全。

        Returns:
            True: プリウォーム成功、False: 接続失敗（グレースフルデグレード）
        """
        if self._prewarmed:
            return True
        try:
            await self.synthesize("テスト")
            self._prewarmed = True
            logger.info("VOICEVOX プリウォーム完了")
            return True
        except httpx.ConnectError:
            logger.warning(
                "VOICEVOX に接続できません (%s)。音声なしで続行します。",
                self.base_url,
            )
            return False
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "VOICEVOX プリウォーム失敗 (%s)。音声なしで続行します。",
                exc.response.status_code,
            )
            return False

    async def synthesize(self, text: str, speaker_id: int | None = None) -> bytes:
        """テキストを音声に変換する。

        Args:
            text: 読み上げテキスト
            speaker_id: 話者ID（省略時はconfig値）

        Returns:
            WAV 形式の音声データ
        """
        speaker = speaker_id or self.speaker_id

        query_response = await self.client.post(
            f"{self.base_url}/audio_query",
            params={"text": text, "speaker": speaker},
        )
        query_response.raise_for_status()
        query = query_response.json()

        query["speedScale"] = config.voicevox.speed_scale
        query["pitchScale"] = config.voicevox.pitch_scale
        query["intonationScale"] = config.voicevox.intonation_scale
        query["volumeScale"] = config.voicevox.volume_scale

        synthesis_response = await self.client.post(
            f"{self.base_url}/synthesis",
            params={"speaker": speaker},
            json=query,
        )
        synthesis_response.raise_for_status()

        return synthesis_response.content

    async def synthesize_to_file(
        self, text: str, output_path: str, speaker_id: int | None = None
    ) -> str:
        """テキストを音声ファイルに保存する。

        Args:
            text: 読み上げテキスト
            output_path: 出力ファイルパス
            speaker_id: 話者ID

        Returns:
            保存したファイルパス
        """
        audio_data = await self.synthesize(text, speaker_id)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_data)

        return str(path)

    async def synthesize_and_play(self, text: str) -> None:
        """テキストを音声合成し再生する。VOICEVOX 停止時・再生失敗時はスキップ。

        Args:
            text: 読み上げテキスト
        """
        try:
            audio_data = await self.synthesize(text)
        except httpx.ConnectError:
            logger.warning("VOICEVOX に接続できません。音声再生をスキップします。")
            return
        except httpx.HTTPStatusError as exc:
            logger.warning("VOICEVOX 合成エラー (%s)。音声再生をスキップします。", exc.response.status_code)
            return

        await self.play(audio_data)

    @staticmethod
    def wav_duration_sec(wav_data: bytes) -> float:
        """WAV の実時間（秒）。"""
        import soundfile as sf  # type: ignore[import-untyped]

        info = sf.info(io.BytesIO(wav_data))
        return float(info.frames) / float(info.samplerate)

    async def play(self, wav_data: bytes) -> None:
        """合成済みの WAV を再生する（再生完了まで待つ）。失敗時はスキップ。

        WAV の実時間 + play_timeout_margin_sec を超えても終わらなければ、
        警告を出してストリームを強制的に閉じ、そのまま戻る。
        再生は専用のデーモンスレッドで行う（戻らなくなっても executor やプロセス終了を塞がない）。
        """
        try:
            duration = self.wav_duration_sec(wav_data)
        except Exception:
            logger.warning("WAV を解釈できません。音声再生をスキップします。", exc_info=True)
            return
        timeout = duration + self._play_timeout_margin

        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        session = _PlaybackSession()

        def worker() -> None:
            error: BaseException | None = None
            try:
                self._play_wav(wav_data, session)
            except BaseException as exc:  # noqa: BLE001 — イベントループ側へそのまま渡す
                error = exc
            try:
                loop.call_soon_threadsafe(_resolve_future, done, error)
            except RuntimeError:
                pass  # イベントループが先に閉じた

        threading.Thread(target=worker, name="tts-play", daemon=True).start()
        try:
            await asyncio.wait_for(done, timeout)
        except TimeoutError:
            logger.warning(
                "音声再生が %.1f秒（WAV %.2f秒 + %.1f秒）で終わらないため、"
                "出力ストリームを強制的に閉じます",
                timeout,
                duration,
                self._play_timeout_margin,
            )
            session.abort()
        except asyncio.CancelledError:
            session.abort()
            raise
        except Exception:
            logger.warning("音声再生に失敗しました。スキップします。", exc_info=True)

    def _play_wav(self, wav_data: bytes, session: _PlaybackSession | None = None) -> None:
        """WAV データを sounddevice で再生する（同期、再生スレッドから呼ぶ）。

        sd.play()/sd.wait()（グローバルストリーム）は使わず、OutputStream を明示的に開いて
        write() で流し切り、stop()/close() する。
        出力デバイスが WAV のサンプルレートに非対応の場合、
        デバイスのネイティブレートに線形補間でリサンプリングする。
        モノラル音声を 2ch 以上のデバイスへ出す場合は列複製でステレオ化する。
        再生失敗時はデフォルトデバイスへフォールバックして再試行する。
        """
        if session is None:
            session = _PlaybackSession()
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
            import soundfile as sf
        except ImportError:
            logger.warning("sounddevice/soundfile が利用できません。音声再生をスキップします。")
            return

        buf = io.BytesIO(wav_data)
        data, samplerate = sf.read(buf, dtype="float32")

        device_id = self.output_device_id
        data, samplerate = self._prepare_audio(sd, data, samplerate, device_id)

        try:
            self._write_stream(sd, data, samplerate, device_id, session)
        except Exception:
            if device_id is not None and not session.aborted.is_set():
                logger.warning(
                    "出力デバイス %d での再生に失敗。デフォルトデバイスにフォールバックします",
                    device_id,
                    exc_info=True,
                )
                # デフォルトデバイス向けに再変換
                buf.seek(0)
                data_fb, sr_fb = sf.read(buf, dtype="float32")
                data_fb, sr_fb = self._prepare_audio(sd, data_fb, sr_fb, None)
                self._write_stream(sd, data_fb, sr_fb, None, session)
            else:
                raise

    def _write_stream(
        self,
        sd: Any,
        data: np.ndarray,
        samplerate: int,
        device_id: int | None,
        session: _PlaybackSession,
    ) -> None:
        """OutputStream を開き、ブロックごとに write() して流し切ってから閉じる。"""
        frames = data if data.ndim == 2 else data.reshape(-1, 1)
        channels = int(frames.shape[1])
        block = max(1, int(samplerate * self._write_block_sec))
        duration = frames.shape[0] / samplerate
        device_label = device_id if device_id is not None else "default"

        stream = sd.OutputStream(
            samplerate=samplerate, channels=channels, dtype="float32", device=device_id,
        )
        session.attach(stream)
        t0 = time.monotonic()
        logger.info("再生開始: device=%s, %.2f秒", device_label, duration)
        try:
            stream.start()
            for start in range(0, frames.shape[0], block):
                if session.aborted.is_set():
                    break
                stream.write(np.ascontiguousarray(frames[start : start + block]))
            if not session.aborted.is_set():
                stream.stop()  # 出力バッファを鳴らし切るまで待つ
        finally:
            session.detach()
            if not session.aborted.is_set():
                stream.close()
        elapsed = time.monotonic() - t0
        if session.aborted.is_set():
            logger.warning(
                "再生中断: device=%s, 経過 %.2f秒（WAV %.2f秒）", device_label, elapsed, duration
            )
        else:
            logger.info(
                "再生終了: device=%s, 経過 %.2f秒（WAV %.2f秒）", device_label, elapsed, duration
            )

    def _prepare_audio(
        self,
        sd: Any,
        data: np.ndarray,
        samplerate: int,
        device_id: int | None,
    ) -> tuple[np.ndarray, int]:
        """再生用にリサンプリングとチャネル変換を行う。

        Returns:
            変換後の (data, samplerate) タプル。
        """
        src_rate = samplerate
        src_channels = 1 if data.ndim == 1 else data.shape[1]
        dst_channels = src_channels

        # --- レート変換 ---
        if device_id is not None:
            try:
                sd.check_output_settings(
                    device=device_id, samplerate=samplerate,
                )
            except sd.PortAudioError:
                dev_info: dict[str, Any] = sd.query_devices(device_id, "output")
                native_rate = int(dev_info["default_samplerate"])
                data = self._resample_audio(data, samplerate, native_rate)
                samplerate = native_rate

        # --- チャネル変換（モノラル→ステレオ） ---
        if device_id is not None:
            dev_info_ch: dict[str, Any] = sd.query_devices(device_id, "output")
            max_ch: int = int(dev_info_ch["max_output_channels"])
            if data.ndim == 1 and max_ch >= 2:
                data = np.column_stack([data, data])
                dst_channels = 2

        logger.info(
            "TTS 再生: device=%s, ch=%d→%d, rate=%d→%d",
            device_id if device_id is not None else "default",
            src_channels,
            dst_channels,
            src_rate,
            samplerate,
        )

        return data, samplerate

    @staticmethod
    def _resample_audio(
        data: np.ndarray, src_rate: int, dst_rate: int,
    ) -> np.ndarray:
        """線形補間でオーディオデータをリサンプリングする。

        モノラル (samples,) およびマルチチャネル (samples, channels) に対応。
        """
        src_len = data.shape[0]
        dst_len = int(src_len * dst_rate / src_rate)
        indices = np.linspace(0, src_len - 1, dst_len)
        idx_floor = indices.astype(np.int64)
        idx_ceil = np.minimum(idx_floor + 1, src_len - 1)
        frac = (indices - idx_floor).astype(np.float32)

        if data.ndim == 1:
            result: np.ndarray = data[idx_floor] * (1 - frac) + data[idx_ceil] * frac
            return result
        # マルチチャネル: frac を列方向にブロードキャスト
        frac = frac[:, np.newaxis]
        result_mc: np.ndarray = data[idx_floor] * (1 - frac) + data[idx_ceil] * frac
        return result_mc

    async def get_speakers(self) -> list[dict[str, Any]]:
        """利用可能な話者一覧を取得する。"""
        response = await self.client.get(f"{self.base_url}/speakers")
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        """クライアントを閉じる。"""
        await self.client.aclose()


async def test_tts() -> None:
    """TTS クライアントのテスト。"""
    client = TTSClient()
    try:
        print("音声合成テスト開始...")

        test_texts = [
            "いらっしゃいませ。ご用件をお伺いできますか？",
            "かしこまりました。総務にお伝えいたしますので、少々お待ちください。",
            "お待たせいたしました。どうぞお入りください。",
        ]

        for i, text in enumerate(test_texts):
            output_path = f"test_tts_{i + 1}.wav"
            await client.synthesize_to_file(text, output_path)
            print(f"生成完了: {output_path}")

        print("\n話者一覧の一部:")
        speakers = await client.get_speakers()
        for speaker in speakers[:3]:
            print(f"  {speaker['name']}: {[s['id'] for s in speaker['styles']]}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_tts())
