"""VOICEVOX TTS クライアント。

受付ボットから移植。synthesize_and_play() によるグレースフルデグレード付き
音声再生機能を追加。
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .config import config

logger = logging.getLogger(__name__)


class TTSClient:
    """VOICEVOX 音声合成クライアント。"""

    def __init__(self, output_device_id: int | None = None) -> None:
        self.base_url = config.voicevox.base_url
        self.speaker_id = config.voicevox.speaker_id
        self.output_device_id = output_device_id
        self.client = httpx.AsyncClient(timeout=30.0)
        self._prewarmed = False

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

    async def play(self, wav_data: bytes) -> None:
        """合成済みの WAV を再生する（再生完了まで待つ）。失敗時はスキップ。"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._play_wav, wav_data)
        except Exception:
            logger.warning("音声再生に失敗しました。スキップします。", exc_info=True)

    def _play_wav(self, wav_data: bytes) -> None:
        """WAV データを sounddevice で再生する（同期）。

        出力デバイスが WAV のサンプルレートに非対応の場合、
        デバイスのネイティブレートに線形補間でリサンプリングする。
        モノラル音声を 2ch 以上のデバイスへ出す場合は列複製でステレオ化する。
        再生失敗時はデフォルトデバイスへフォールバックして再試行する。
        """
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("sounddevice/soundfile が利用できません。音声再生をスキップします。")
            return

        buf = io.BytesIO(wav_data)
        data, samplerate = sf.read(buf, dtype="float32")

        device_id = self.output_device_id
        data, samplerate = self._prepare_audio(sd, data, samplerate, device_id)

        try:
            sd.play(data, samplerate, device=device_id)
            sd.wait()
        except Exception:
            if device_id is not None:
                logger.warning(
                    "出力デバイス %d での再生に失敗。デフォルトデバイスにフォールバックします",
                    device_id,
                    exc_info=True,
                )
                # デフォルトデバイス向けに再変換
                buf.seek(0)
                data_fb, sr_fb = sf.read(buf, dtype="float32")
                data_fb, sr_fb = self._prepare_audio(sd, data_fb, sr_fb, None)
                sd.play(data_fb, sr_fb, device=None)
                sd.wait()
            else:
                raise

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
