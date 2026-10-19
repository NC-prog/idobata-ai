"""井戸端AI 音声往復 CLI（WP00）。

- ``--mode text`` : 標準入力の 1 行 → dialogue → 応答表示 + VOICEVOX 再生
- ``--mode voice``: マイク → SpeechGate → VAD → ASR → dialogue → VOICEVOX 再生
- ``--mode voice --input-wav a.wav b.wav ...``: マイクの代わりに WAV を順に流す疑似音声モード。
  ``--output-wav-dir`` を付けると再生せず応答音声を WAV で書き出す（無人での自動確認用）。

会話ログは保存しない。発話・応答の本文は DEBUG レベルでのみログに出す。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import numpy as np

from .asr_client import ASRClient, load_site_vocabulary
from .audio_input import MicrophoneInput, WavFileInput, list_audio_devices
from .config import ConfigError, config
from .dialogue import Dialogue
from .llm_client import LLMClient
from .speech_gate import SpeechGate
from .tts_client import TTSClient
from .vad import SpeechEventType, VADBackend, create_vad

logger = logging.getLogger(__name__)

# WAV 入力の末尾に足す無音（VAD に SPEECH_END を出させるため）
_WAV_TAIL_SILENCE_SEC = 1.0
# 疑似音声モードでゲートが開くのを待つときの確認間隔
_GATE_POLL_SEC = 0.05


@dataclass
class TurnTiming:
    """1 往復のレイテンシ計測（monotonic 秒）。"""

    t_speech_end: float
    t_asr_done: float = 0.0
    t_llm_done: float = 0.0
    t_audio_start: float = 0.0

    def summary(self) -> str:
        """発話終了（VAD 判定）を起点とした各段の経過時間。"""
        base = self.t_speech_end
        return (
            f"ASR {self.t_asr_done - base:.2f}s / "
            f"LLM {self.t_llm_done - base:.2f}s / "
            f"音声開始 {self.t_audio_start - base:.2f}s"
        )


class VoiceLoop:
    """音声往復アプリケーション。"""

    def __init__(
        self,
        mode: str,
        input_wavs: list[Path],
        audio_device: int | None,
        output_device: int | None,
        output_wav_dir: Path | None,
    ) -> None:
        self._mode = mode
        self._input_wavs = input_wavs
        # CLI 指定がなければ .env（IDOBATA_AUDIO_INPUT_DEVICE / _OUTPUT_DEVICE）の値
        self._audio_device = (
            audio_device if audio_device is not None else config.audio_input.device_id
        )
        self._output_wav_dir = output_wav_dir
        self._tts = TTSClient(
            output_device_id=(
                output_device if output_device is not None else config.audio_output.device_id
            )
        )
        self._dialogue = Dialogue(LLMClient())
        self._gate = SpeechGate(
            post_margin_sec=config.speech_gate.post_margin_sec,
            safety_margin_sec=config.speech_gate.safety_margin_sec,
        )
        self._asr: ASRClient | None = None
        self._reply_count = 0
        self._turn_lock = asyncio.Lock()
        self._turn_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # 起動
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """モードに応じたループを実行する。"""
        await self._tts.prewarm()
        self._print_banner()
        try:
            if self._mode == "text":
                await self._text_loop()
            else:
                await self._voice_loop()
        finally:
            await self._shutdown()

    def _print_banner(self) -> None:
        print("=" * 50)
        print("  井戸端AI — WP00 音声往復")
        print(f"  プロファイル: {config.profile.value} / 拠点: {config.site}")
        print(f"  モード: {self._mode}")
        print(f"  LLM: {config.llm.gateway_url} ({config.llm.chat_model})")
        if self._mode == "voice":
            print(f"  ASR: {config.asr.base_url}")
            if not self._input_wavs:
                print(f"  音声入力デバイス: [{self._audio_device}]")
            if self._input_wavs:
                print(f"  入力WAV: {', '.join(p.name for p in self._input_wavs)}")
        if self._output_wav_dir is not None:
            print(f"  応答音声の書き出し先: {self._output_wav_dir}")
        elif self._tts.output_device_id is not None:
            print(f"  音声出力デバイス: [{self._tts.output_device_id}]")
        print("=" * 50)
        if self._mode == "text":
            print("話しかける内容を入力してください（/reset で履歴消去、/quit で終了）")
        print()

    # ------------------------------------------------------------------
    # text モード
    # ------------------------------------------------------------------

    async def _text_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                raw = await loop.run_in_executor(None, input, "> ")
            except EOFError:
                break
            text = raw.strip()
            if not text:
                continue
            if text in ("/quit", "/q", "/exit"):
                break
            if text == "/reset":
                self._dialogue.reset()
                print("[履歴を消去しました]")
                continue
            await self._respond_and_speak(text, "ja", None)

    # ------------------------------------------------------------------
    # voice モード
    # ------------------------------------------------------------------

    async def _voice_loop(self) -> None:
        self._asr = ASRClient()
        self._asr.set_static_context(
            load_site_vocabulary(config.site, config.asr.site_context_dir)
        )

        # VAD モデルを先にロード（マイク起動前にロードしないとフレームドロップする）
        vad = create_vad(config.vad)
        logger.info("VAD モデルロード完了")

        if self._input_wavs:
            for wav in self._input_wavs:
                source = WavFileInput(wav, config.audio_input)
                await source.start()
                await self._run_pipeline(vad, _with_tail_silence(source))
                await source.stop()
                # 疑似音声では 1 ファイル = 1 往復として、応答が終わりゲートが開くまで待つ
                # （閉じたまま次のファイルを流すと、冒頭がゲートで捨てられる）
                await self._drain_turns()
                await self._wait_gate_open()
            return

        mic_cfg = replace(config.audio_input, device_id=self._audio_device)
        mic = MicrophoneInput(mic_cfg)
        await mic.start()
        print("マイク入力中。話しかけてください（Ctrl+C で終了）")
        try:
            await self._run_pipeline(vad, mic)
        finally:
            await mic.stop()

    async def _run_pipeline(self, vad: VADBackend, frames: AsyncIterator[np.ndarray]) -> None:
        """音声フレーム → (SpeechGate) → VAD → ASR → 応答タスク。"""
        assert self._asr is not None
        segment_frames: list[np.ndarray] = []
        t_speech_start = 0.0
        was_muted = False

        async for frame in frames:
            # --- 自己発話ゲート（townhall と同じ） ---
            if self._gate.is_muted:
                if not was_muted:
                    if vad.in_speech:
                        logger.info(
                            "[GATE] VAD 発話中にミュート開始 — セグメント破棄 (%d フレーム)",
                            len(segment_frames),
                        )
                        vad.reset()
                    segment_frames = []
                    was_muted = True
                continue
            if was_muted:
                was_muted = False
            # --- /自己発話ゲート ---

            event = vad.process_frame(frame)
            if event is not None:
                if event.event_type == SpeechEventType.SPEECH_START:
                    segment_frames = [frame]
                    t_speech_start = event.timestamp_sec
                elif event.event_type == SpeechEventType.SPEECH_END:
                    if segment_frames:
                        await self._on_segment(segment_frames, t_speech_start)
                    segment_frames = []
            elif vad.in_speech:
                segment_frames.append(frame)

    async def _on_segment(self, frames: list[np.ndarray], t_speech_start: float) -> None:
        """1 セグメントを ASR に掛け、確定テキストが出たら応答タスクを起こす。"""
        assert self._asr is not None
        timing = TurnTiming(t_speech_end=time.monotonic())
        final_text = ""
        language = "ja"
        async for chunk in self._asr.transcribe_segment(
            frames, t_speech_start, sample_rate=config.vad.sample_rate
        ):
            if chunk.is_final:
                final_text = chunk.text
                language = chunk.language
        timing.t_asr_done = time.monotonic()
        if not final_text:
            return

        self._asr.push_rolling_context(final_text)
        print(f"[聞き取り:{language}] {final_text}")
        # 応答生成中も音声フレームの消費を止めない（マイクのキューが詰まらないように）
        task = asyncio.create_task(
            self._respond_and_speak(final_text, language, timing), name="turn"
        )
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _drain_turns(self) -> None:
        while self._turn_tasks:
            await asyncio.gather(*list(self._turn_tasks), return_exceptions=True)

    async def _wait_gate_open(self) -> None:
        # 再生が戻らなくてもゲートの安全装置で開くので、無限には待たない
        while self._gate.is_muted:
            await asyncio.sleep(_GATE_POLL_SEC)

    # ------------------------------------------------------------------
    # 応答 → 発話
    # ------------------------------------------------------------------

    async def _respond_and_speak(
        self, text: str, language: str, timing: TurnTiming | None
    ) -> None:
        # 応答は 1 つずつ順に話す（重なり防止）
        async with self._turn_lock:
            reply = await self._dialogue.respond(text, language)
            if timing is not None:
                timing.t_llm_done = time.monotonic()
            if not reply:
                return
            print(f"[井戸端AI] {reply}")
            await self._speak(reply, timing)
            if timing is not None:
                print(f"[計測] 発話終了→ {timing.summary()}")

    async def _speak(self, text: str, timing: TurnTiming | None) -> None:
        """VOICEVOX で合成し、再生（または WAV 書き出し）する。再生中はゲートを閉じる。

        合成後に想定再生時間を渡してゲートの安全装置を張る。再生が戻らなくても
        「再生時間 + 後マージン + 安全マージン」でゲートは開く。
        """
        self._gate.mute()
        try:
            try:
                wav = await self._tts.synthesize(text)
            except (httpx.ConnectError, httpx.HTTPStatusError):
                logger.warning("VOICEVOX 合成に失敗。音声なしで続行します。")
                return
            try:
                self._gate.mute(expected_sec=TTSClient.wav_duration_sec(wav))
            except Exception:
                logger.warning("合成音声の長さを読めません。再生をスキップします。", exc_info=True)
                return
            # 合成完了 = 再生（または書き出し）開始とみなす
            if timing is not None:
                timing.t_audio_start = time.monotonic()
            if self._output_wav_dir is not None:
                self._reply_count += 1
                out = self._output_wav_dir / f"reply_{self._reply_count:02d}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(wav)
                print(f"[音声] {out}")
            else:
                await self._tts.play(wav)
        finally:
            self._gate.schedule_unmute()

    async def _shutdown(self) -> None:
        await self._drain_turns()
        if self._asr is not None:
            await self._asr.close()
        await self._tts.close()


async def _with_tail_silence(source: WavFileInput) -> AsyncIterator[np.ndarray]:
    """WAV のフレームに続けて無音フレームを流す（末尾の発話を VAD に確定させる）。"""
    async for frame in source:
        yield frame
    cfg = config.audio_input
    n = int(_WAV_TAIL_SILENCE_SEC * cfg.sample_rate / cfg.frame_size)
    silence = np.zeros(cfg.frame_size, dtype=np.float32)
    for _ in range(n):
        await asyncio.sleep(cfg.frame_size / cfg.sample_rate / cfg.wav_speed)
        yield silence


def main() -> None:
    """エントリポイント。"""
    parser = argparse.ArgumentParser(description="井戸端AI 音声往復 CLI（WP00）")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument(
        "--input-wav", type=Path, nargs="+", default=[],
        help="voice モードでマイクの代わりに流す WAV（複数可、1 ファイル = 1 往復）",
    )
    parser.add_argument(
        "--output-wav-dir", type=Path, default=None,
        help="応答音声を再生せず WAV で書き出すディレクトリ（自動確認用）",
    )
    parser.add_argument(
        "--audio-device", type=int, default=None,
        help="音声入力デバイス ID（省略時は .env の IDOBATA_AUDIO_INPUT_DEVICE）",
    )
    parser.add_argument(
        "--output-device", type=int, default=None,
        help="TTS 音声出力デバイス ID（省略時は .env の IDOBATA_AUDIO_OUTPUT_DEVICE）",
    )
    parser.add_argument("--list-devices", action="store_true", help="音声デバイス一覧を表示して終了")
    parser.add_argument("--debug", action="store_true", help="DEBUG ログ（発話本文を含む）")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpx2", "openai", "websockets", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.input_wav and args.mode != "voice":
        parser.error("--input-wav は --mode voice と一緒に指定してください")

    app = VoiceLoop(
        mode=args.mode,
        input_wavs=args.input_wav,
        audio_device=args.audio_device,
        output_device=args.output_device,
        output_wav_dir=args.output_wav_dir,
    )
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        print(f"[設定エラー] {exc}", file=sys.stderr)
        sys.exit(2)
