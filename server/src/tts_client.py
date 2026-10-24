"""VOICEVOX TTS クライアント。

受付ボットから移植。synthesize_and_play() によるグレースフルデグレード付き
音声再生機能を追加。
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .config import DeviceSpec, config
from .site_context import SiteWord, load_spoken_replacements

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


def _dict_word_differs(
    current: dict[str, Any], reading: str, accent: int, priority: int
) -> bool:
    """登録済みの語（GET /user_dict の値）が、登録したい内容と違うか。"""
    return (
        current.get("pronunciation") != reading
        or current.get("accent_type") != accent
        or current.get("priority") != priority
    )


class TTSClient:
    """VOICEVOX 音声合成クライアント。"""

    def __init__(self, output_device_id: DeviceSpec = None) -> None:
        self.base_url = config.voicevox.base_url
        self.speaker_id = config.voicevox.speaker_id
        self.output_device_id = output_device_id
        self.client = httpx.AsyncClient(timeout=config.voicevox.timeout)
        self._prewarmed = False
        self._play_timeout_margin = config.audio_output.play_timeout_margin_sec
        self._write_block_sec = config.audio_output.write_block_sec
        # 合成の直前に置き換える表記（辞書が効かない記号「℃」→「度」。allowed_terms.yaml の spoken。WP13）
        self.spoken_replacements: list[tuple[str, str]] = load_spoken_replacements(config.dialogue.allowed_terms_path)

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
        wav, _ = await self.synthesize_with_query(text, speaker_id)
        return wav

    async def synthesize_with_query(
        self, text: str, speaker_id: int | None = None
    ) -> tuple[bytes, dict[str, Any]]:
        """テキストを音声に変換し、合成に使った audio_query も返す（口パクの時系列用）。"""
        speaker = speaker_id or self.speaker_id
        cfg = config.voicevox
        for old, new in self.spoken_replacements:
            text = text.replace(old, new)  # 辞書が効かない記号（「℃」→「度」。allowed_terms.yaml の spoken）

        query_response = await self._request(
            "POST", "/audio_query", params={"text": text, "speaker": speaker},
        )
        query: dict[str, Any] = query_response.json()

        query["speedScale"] = cfg.speed_scale
        query["pitchScale"] = cfg.pitch_scale
        query["intonationScale"] = cfg.intonation_scale
        query["volumeScale"] = cfg.volume_scale
        query["postPhonemeLength"] = cfg.post_phoneme_length

        synthesis_response = await self._request(
            "POST", "/synthesis", params={"speaker": speaker}, json=query,
        )
        return synthesis_response.content, query

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """VOICEVOX への HTTP 呼び出し。接続失敗・タイムアウト・5xx は再試行する。"""
        cfg = config.voicevox
        for attempt in range(cfg.max_retries + 1):
            last = attempt == cfg.max_retries
            try:
                response = await self.client.request(method, f"{self.base_url}{path}", **kwargs)
            except httpx.TransportError:
                if last:
                    raise
                logger.warning("VOICEVOX %s に接続できません。再試行 %d/%d",
                               path, attempt + 1, cfg.max_retries)
            else:
                if response.status_code < 500 or last:
                    response.raise_for_status()
                    return response
                logger.warning("VOICEVOX %s が %d を返しました。再試行 %d/%d",
                               path, response.status_code, attempt + 1, cfg.max_retries)
            await asyncio.sleep(cfg.retry_wait_sec)
        raise AssertionError("unreachable")

    async def sync_user_dict(self, words: list[SiteWord]) -> int:
        """読み付きの語を VOICEVOX のユーザー辞書に登録する（同じ表記があれば更新）。

        Returns:
            追加・更新した語の数。VOICEVOX に繋がらない場合は 0（音声は既定の読みで続行）。
        """
        cfg = config.voicevox
        targets = [w for w in words if w.reading]
        if not cfg.user_dict_enabled or not targets:
            return 0
        try:
            existing: dict[str, dict[str, Any]] = (await self._request("GET", "/user_dict")).json()
        except httpx.HTTPError:
            logger.warning("VOICEVOX のユーザー辞書を取得できません。読みの登録をスキップします。")
            return 0
        # VOICEVOX は表記を全角に正規化して保存するので、NFKC で揃えて突き合わせる
        by_surface = {
            unicodedata.normalize("NFKC", str(v.get("surface", ""))): (uuid, v)
            for uuid, v in existing.items()
        }

        changed = 0
        for word in targets:
            accent = word.accent if word.accent is not None else cfg.user_dict_default_accent
            params: dict[str, str | int] = {
                "surface": word.surface,
                "pronunciation": word.reading or "",
                "accent_type": accent,
                "word_type": cfg.user_dict_word_type,
                "priority": cfg.user_dict_priority,
            }
            current = by_surface.get(unicodedata.normalize("NFKC", word.surface))
            try:
                if current is None:
                    await self._request("POST", "/user_dict_word", params=params)
                elif _dict_word_differs(current[1], word.reading or "", accent, cfg.user_dict_priority):
                    await self._request("PUT", f"/user_dict_word/{current[0]}", params=params)
                else:
                    continue
            except httpx.HTTPError:
                logger.warning("ユーザー辞書への登録に失敗: %s", word.surface, exc_info=True)
                continue
            changed += 1
        logger.info("VOICEVOX ユーザー辞書: %d 語中 %d 語を登録・更新", len(targets), changed)
        return changed

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
                    "出力デバイス %s での再生に失敗。デフォルトデバイスにフォールバックします",
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
        device_id: DeviceSpec,
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
        device_id: DeviceSpec,
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

    def open_player(self) -> StreamPlayer:
        """1 応答ぶん（複数の文）を切れ目なく再生するプレイヤーを作る。"""
        return StreamPlayer(self)

    async def get_speakers(self) -> list[dict[str, Any]]:
        """利用可能な話者一覧を取得する。"""
        response = await self.client.get(f"{self.base_url}/speakers")
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        """クライアントを閉じる。"""
        await self.client.aclose()


class StreamPlayer:
    """複数の WAV（1 文ずつ）を 1 本の OutputStream で切れ目なく再生する。

    1 つ目の push() で再生スレッドを起こしてストリームを開き、以降の WAV は届き次第
    続けて書く。次の WAV がまだ無い間は短い無音を書いてストリームを止めない
    （開き直すと音が途切れ、ALSA ではアンダーランの雑音にもなる）。
    finish() で「最後まで鳴らし切る」のを待つ。待ち時間は再生終了の見込み + 余裕で打ち切る。
    """

    def __init__(self, tts: TTSClient) -> None:
        self._tts = tts
        self._queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self._session = _PlaybackSession()
        self._done: asyncio.Future[None] | None = None
        self._end_at = 0.0  # 再生終了の見込み（monotonic）

    def push(self, wav: bytes) -> float:
        """1 文ぶんの WAV を再生キューに積む（初回で再生開始）。

        Returns:
            今から再生終了までの見込み秒数（ゲートの安全装置に渡す）。
        """
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav), dtype="float32")
        now = time.monotonic()
        self._end_at = max(now, self._end_at) + len(data) / sr
        if self._done is None:
            loop = asyncio.get_running_loop()
            self._done = loop.create_future()
            threading.Thread(
                target=self._worker, args=(loop, self._done, (data, int(sr))),
                name="tts-stream", daemon=True,
            ).start()
        else:
            self._queue.put((data, int(sr)))
        return self._end_at - now

    async def finish(self) -> None:
        """積んだ WAV を鳴らし切るまで待つ。失敗・タイムアウト時は警告してストリームを閉じる。"""
        if self._done is None:
            return
        self._queue.put(None)
        timeout = max(0.0, self._end_at - time.monotonic()) + self._tts._play_timeout_margin
        try:
            await asyncio.wait_for(asyncio.shield(self._done), timeout)
        except TimeoutError:
            logger.warning("連続再生が見込み + %.1f秒で終わらないため、出力ストリームを閉じます",
                           self._tts._play_timeout_margin)
            self._session.abort()
        except asyncio.CancelledError:
            self._session.abort()
            raise
        except Exception:
            logger.warning("連続再生に失敗しました。スキップします。", exc_info=True)

    def _worker(
        self,
        loop: asyncio.AbstractEventLoop,
        done: asyncio.Future[None],
        first: tuple[np.ndarray, int],
    ) -> None:
        error: BaseException | None = None
        try:
            self._run(first)
        except BaseException as exc:  # noqa: BLE001 — イベントループ側へそのまま渡す
            error = exc
        try:
            loop.call_soon_threadsafe(_resolve_future, done, error)
        except RuntimeError:
            pass  # イベントループが先に閉じた

    def _run(self, first: tuple[np.ndarray, int]) -> None:
        import sounddevice as sd

        device = self._tts.output_device_id
        try:
            stream, data, rate = self._open(sd, first, device)
        except Exception:
            if device is None:
                raise
            logger.warning("出力デバイス %s を開けません。デフォルトデバイスにフォールバックします",
                           device, exc_info=True)
            device = None
            stream, data, rate = self._open(sd, first, device)

        channels = int(stream.channels)
        fill = np.zeros((max(1, int(rate * config.audio_output.idle_fill_sec)), channels),
                        dtype=np.float32)
        t0 = time.monotonic()
        total = 0.0
        logger.info("連続再生開始: device=%s", device if device is not None else "default")
        try:
            self._write(stream, data, rate)
            total += data.shape[0] / rate
            while not self._session.aborted.is_set():
                try:
                    item = self._queue.get(timeout=config.audio_output.idle_fill_sec)
                except queue.Empty:
                    stream.write(fill)  # 次の文の合成待ち。無音で繋ぐ
                    continue
                if item is None:
                    break
                chunk = self._conform(item, rate, channels)
                self._write(stream, chunk, rate)
                total += chunk.shape[0] / rate
            if not self._session.aborted.is_set():
                stream.stop()  # 出力バッファを鳴らし切るまで待つ
        finally:
            self._session.detach()
            if not self._session.aborted.is_set():
                stream.close()
        logger.info("連続再生終了: 経過 %.2f秒（音声 %.2f秒）", time.monotonic() - t0, total)

    def _open(
        self, sd: Any, first: tuple[np.ndarray, int], device: DeviceSpec
    ) -> tuple[Any, np.ndarray, int]:
        """1 つ目の WAV に合わせてデバイス向けの形式を決め、ストリームを開いて始める。"""
        data, rate = self._tts._prepare_audio(sd, first[0], first[1], device)
        frames = data if data.ndim == 2 else data.reshape(-1, 1)
        stream = sd.OutputStream(
            samplerate=rate, channels=int(frames.shape[1]), dtype="float32", device=device,
        )
        self._session.attach(stream)
        stream.start()
        return stream, frames, rate

    def _conform(self, item: tuple[np.ndarray, int], rate: int, channels: int) -> np.ndarray:
        """2 つ目以降の WAV を、開いたストリームのレート・チャネル数に合わせる。"""
        data, sr = item
        if sr != rate:
            data = TTSClient._resample_audio(data, sr, rate)
        frames = data if data.ndim == 2 else data.reshape(-1, 1)
        if frames.shape[1] != channels:
            frames = np.repeat(frames[:, :1], channels, axis=1)
        return frames.astype(np.float32, copy=False)

    def _write(self, stream: Any, frames: np.ndarray, rate: int) -> None:
        block = max(1, int(rate * self._tts._write_block_sec))
        for start in range(0, frames.shape[0], block):
            if self._session.aborted.is_set():
                return
            stream.write(np.ascontiguousarray(frames[start : start + block]))


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
