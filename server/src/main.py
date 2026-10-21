"""井戸端AI 音声往復 CLI（WP00 / WP01）。

- ``--mode text`` : 標準入力の 1 行 → dialogue → 応答表示 + VOICEVOX 再生
- 応答は LLM のストリーミングを文に切り、1 文目から合成・再生する（WP01）
- ``--mode voice``: マイク → SpeechGate → VAD → ASR → dialogue → VOICEVOX 再生
- ``--edge``: マイク・スピーカー・アバターをブラウザ（エッジ）に置く。音声は WebSocket で
  受け取り、応答は文ごとに口パク付きでエッジへ送る（``http://localhost:8080/edge/``）
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
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .actions import Conversation
from .asr_client import ASRClient, language_name, load_site_vocabulary
from .audio_input import EdgeAudioInput, MicrophoneInput, WavFileInput, list_audio_devices
from .backchannel import BackchannelBank, BackchannelClip, strip_leading_backchannel
from .config import SERVER_ROOT, ConfigError, DeviceSpec, config, parse_device
from .dialogue import Dialogue
from .edge_bridge import EdgeBridge
from .edge_server import EdgeServer, StaffPanel
from .knowledge import KnowledgeStore, Weather
from .llm_client import LLMClient
from .lookup import Lookup, close_browser
from .procedures import AsrFixes, load_procedures
from .referrals import ReferralStore, TidbitStore, VisitLog, load_tags
from .site_context import load_facility, load_site_words, referral_csv_enabled
from .speech_gate import SpeechGate
from .speech_output import (
    EdgeSpeaker,
    LocalSpeaker,
    SpeechSink,
    SpokenSentence,
    WavDirWriter,
)
from .timeofday import greeting_for
from .tts_client import TTSClient
from .vad import SpeechEventType, VADBackend, create_vad

logger = logging.getLogger(__name__)

# WAV 入力の末尾に足す無音（VAD に SPEECH_END を出させるため）
_WAV_TAIL_SILENCE_SEC = 1.0
# 疑似音声モードでゲートが開くのを待つときの確認間隔
_GATE_POLL_SEC = 0.05
# 待機タイムアウト（最後の確定発話からの経過）を確認する間隔
_IDLE_POLL_SEC = 1.0


@dataclass
class TurnTiming:
    """1 往復のレイテンシ計測（monotonic 秒）。"""

    t_speech_end: float
    t_asr_done: float = 0.0
    t_backchannel: float = 0.0  # 相槌の再生開始（WP04）
    t_first_sentence: float = 0.0
    t_audio_start: float = 0.0
    t_llm_done: float = 0.0
    search_ms: float | None = None  # 知識パックの検索（LLM の前）

    def summary(self) -> str:
        """発話終了（VAD 判定）を起点とした各段の経過時間。"""
        base = self.t_speech_end

        def at(t: float) -> str:
            return f"{t - base:.2f}s" if t else "-"

        return (
            f"ASR {at(self.t_asr_done)} / "
            f"相槌開始 {at(self.t_backchannel)} / "
            f"検索 {'-' if self.search_ms is None else f'{self.search_ms:.0f}ms'} / "
            f"LLM 1文目確定 {at(self.t_first_sentence)} / "
            f"音声開始 {at(self.t_audio_start)} / "
            f"LLM 完了 {at(self.t_llm_done)}"
        )


class VoiceLoop:
    """音声往復アプリケーション。"""

    def __init__(
        self,
        mode: str,
        input_wavs: list[Path],
        audio_device: DeviceSpec,
        output_device: DeviceSpec,
        output_wav_dir: Path | None,
        edge: bool = False,
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
        # 拠点の固定事実（facility）はプロンプトに直書き。知識パックは run() で読む（数秒かかる）
        llm = LLMClient()
        facility = load_facility(config.site, config.asr.site_context_dir)
        self._dialogue = Dialogue(
            llm,
            facility=facility,
            weather=Weather.for_site(config.data_dir, config.site),
            avatar_name=config.actions.avatar_name,
            enabled_actions=config.actions.enabled,
        )
        # 同意付きの行為（調べる・つなぐ）と来訪記録（WP04）
        self._referrals = ReferralStore(config.data_dir, config.site, config.referral.retention_days,
                                        csv_export=referral_csv_enabled(config.site))
        self._visits = VisitLog(config.data_dir, config.site)
        self._tidbits = TidbitStore(config.data_dir, config.site, config.tidbit.retention_days)
        self._asr_fixes = AsrFixes(config.asr.site_context_dir, config.data_dir)
        self._lookup = Lookup(config.lookup, config.site, config.data_dir, self._make_lookup_query)
        self._conversation = Conversation(
            self._dialogue, llm, config.llm.chat_model, config.actions, config.referral, facility,
            load_tags(SERVER_ROOT / "context" / "tags.yaml"),
            lookup=self._lookup,
            referrals=self._referrals, visits=self._visits,
            on_state=self._send_state, edge_id=self._edge_id,
            tidbits=self._tidbits, tidbit_cfg=config.tidbit,
            procedures=load_procedures(config.asr.site_context_dir),
        )
        self._llm = llm
        self._backchannels = (BackchannelBank(config.actions.backchannel_path)
                              if config.actions.backchannel_enabled else None)
        self._gate = SpeechGate(
            post_margin_sec=config.speech_gate.post_margin_sec,
            safety_margin_sec=config.speech_gate.safety_margin_sec,
        )
        self._asr: ASRClient | None = None
        self._reply_count = 0
        self._turn_lock = asyncio.Lock()
        self._turn_tasks: set[asyncio.Task[None]] = set()
        # --edge: マイク・スピーカー・アバターをブラウザ（エッジ）側に置く
        self._edge: EdgeBridge | None = None
        self._edge_server: EdgeServer | None = None
        if edge:
            self._edge = EdgeBridge(config.edge.ws_host, config.edge.ws_port)
        # エッジのページと職員パネル（/staff/）は同じ HTTP サーバ。職員パネルだけでも立てる
        if edge or config.staff.password:
            self._edge_server = EdgeServer(
                config.edge.http_host, config.edge.http_port, config.edge.static_dir,
                staff=StaffPanel(self._referrals, self._visits, config.staff, self._tidbits,
                                 tuple(d.name for d in (facility.departments if facility else ())),
                                 self._asr_fixes),
            )
        self._bg_tasks: set[asyncio.Task[None]] = set()
        # 知識パック（毎朝作り直され、再起動なしで差し替わる）
        self._knowledge = KnowledgeStore(config.data_dir, config.site, config.knowledge)
        # 最後に確定した来訪者の発話の時刻（monotonic）。待機に戻ったら None
        self._last_utterance_at: float | None = None

    # ------------------------------------------------------------------
    # 起動
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """モードに応じたループを実行する。"""
        # 固有名詞の読みを先に登録する（プリウォームの合成より前に）
        await self._tts.sync_user_dict(load_site_words(config.site, config.asr.site_context_dir))
        await self._tts.prewarm()
        if self._backchannels is not None:
            await self._backchannels.prepare(self._tts, config.actions.backchannel_prepare_budget_sec)
        if "lookup" in config.actions.enabled:
            await self._lookup.prepare()
        removed = self._referrals.cleanup() + self._visits.cleanup() + self._tidbits.cleanup()
        if removed:
            logger.info("保存期間を過ぎた紹介状・来訪記録・小ネタを %d 件消しました", removed)
        await asyncio.to_thread(self._knowledge.reload_if_changed)
        self._dialogue.attach_knowledge(self._knowledge)
        self._spawn(self._watch_knowledge(), "kb-watch")
        if self._edge is not None:
            self._edge.on_hello = self._on_edge_hello
            await self._edge.start()
        if self._edge_server is not None:
            await self._edge_server.start()
        self._print_banner()
        idle_watch = asyncio.create_task(self._watch_idle(), name="idle-watch")
        try:
            if self._mode == "text":
                await self._text_loop()
            else:
                await self._voice_loop()
        finally:
            idle_watch.cancel()
            await self._shutdown()

    def _print_banner(self) -> None:
        print("=" * 50)
        print("  井戸端AI — WP01 音声往復")
        print(f"  プロファイル: {config.profile.value} / 拠点: {config.site}")
        print(f"  モード: {self._mode}")
        print(f"  LLM: {config.llm.gateway_url} ({config.llm.chat_model})")
        if self._mode == "voice":
            print(f"  ASR: {config.asr.base_url}")
            if self._edge is not None and not self._input_wavs:
                print("  音声入力: エッジ（ブラウザのマイク）")
            elif not self._input_wavs:
                print(f"  音声入力デバイス: [{self._audio_device}]")
            if self._input_wavs:
                print(f"  入力WAV: {', '.join(p.name for p in self._input_wavs)}")
        if self._edge_server is not None:
            print(f"  エッジ: {self._edge_server.url} を Chromium で開く"
                  f"（WebSocket ws://{config.edge.ws_host}:{config.edge.ws_port}）")
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
                await self._conversation.end_session()
                print("[会話を区切りました]")
                continue
            self._last_utterance_at = time.monotonic()
            await self._send_caption(self._shown_user_text(text), role="user")
            await self._send_state("thinking")
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

        if self._edge is not None:
            edge_input = EdgeAudioInput(config.audio_input)
            self._edge.on_pcm = edge_input.push_pcm
            print("エッジの接続を待っています。ブラウザで画面を開き「はじめる」を押してください"
                  "（Ctrl+C で終了）")
            try:
                await self._run_pipeline(vad, edge_input)
            finally:
                self._edge.on_pcm = None
                await edge_input.stop()
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
        preroll_len = max(1, config.vad.segment_preroll_ms * config.vad.sample_rate
                          // (1000 * config.vad.frame_size))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_len)

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
                    preroll.clear()  # ミュート前の音（自分の声の直前）を次の発話に混ぜない
                    was_muted = True
                continue
            if was_muted:
                was_muted = False
            # --- /自己発話ゲート ---

            event = vad.process_frame(frame)
            if event is not None:
                if event.event_type == SpeechEventType.SPEECH_START:
                    segment_frames = [*preroll, frame]
                    preroll.clear()
                    t_speech_start = event.timestamp_sec
                    await self._send_state("listening")
                elif event.event_type == SpeechEventType.SPEECH_END:
                    if segment_frames:
                        await self._on_segment(segment_frames, t_speech_start)
                    segment_frames = []
            elif vad.in_speech:
                segment_frames.append(frame)
            else:
                preroll.append(frame)

    async def _on_segment(self, frames: list[np.ndarray], t_speech_start: float) -> None:
        """1 セグメントを ASR に掛け、確定テキストが出たら応答タスクを起こす。"""
        assert self._asr is not None
        timing = TurnTiming(t_speech_end=time.monotonic())
        await self._send_state("thinking")
        final_text = ""
        language = "ja"
        # 提案の返事待ちは短い発話（うん・はい）が多く、言語判定を誤りやすいので会話の言語に固定する
        force = (language_name(self._conversation.language)
                 if config.asr.force_language_on_reply and self._conversation.expects_short_reply
                 else None)
        async for chunk in self._asr.transcribe_segment(
            frames, t_speech_start, sample_rate=config.vad.sample_rate, force_language=force
        ):
            if chunk.is_final:
                final_text = chunk.text
                language = chunk.language
        timing.t_asr_done = time.monotonic()
        if not final_text:
            await self._send_state("idle")
            return

        self._last_utterance_at = time.monotonic()
        self._asr.push_rolling_context(final_text)
        final_text = self._asr_fixes.apply(final_text)
        shown = self._shown_user_text(final_text)
        print(f"[聞き取り:{language}] {shown}")
        await self._send_caption(shown, role="user")
        # 応答生成中も音声フレームの消費を止めない（マイクのキューが詰まらないように）
        task = asyncio.create_task(
            self._respond_and_speak(final_text, language, timing), name="turn"
        )
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _watch_idle(self) -> None:
        """最後の確定発話から idle_timeout_sec 経ったら、履歴を消して待機に戻る。"""
        timeout = config.dialogue.idle_timeout_sec
        if timeout <= 0:
            return
        while True:
            await asyncio.sleep(_IDLE_POLL_SEC)
            await self.check_idle(time.monotonic(), timeout)

    async def check_idle(self, now: float, timeout: float) -> bool:
        """待機タイムアウトに達していれば会話を区切る。区切ったら True。"""
        last = self._last_utterance_at
        if last is None or now - last < timeout or self._turn_lock.locked():
            return False  # 応答中は区切らない（終わってから次の確認で区切る）
        self._last_utterance_at = None
        await self._conversation.end_session()  # 来訪記録を 1 行書いて、履歴を消す
        if self._asr is not None:
            self._asr.clear_rolling_context()
        logger.info("最後の発話から %.0f秒経ったため会話を区切りました（次の発話から新しい会話）",
                    timeout)
        await self._send_state("idle")
        return True

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
        """LLM の応答を 1 文ずつ受け、1 文できるごとに合成して流す。

        LLM（文の生成）と合成・再生は並行に進む。合成は直列（VOICEVOX は CPU で十分速い）。
        ゲートは 1 文目を流す直前に閉じ、最後の文を鳴らし終えてから後マージン後に開く。
        """
        # 応答は 1 つずつ順に話す（重なり防止）
        async with self._turn_lock:
            sentences: asyncio.Queue[str | None] = asyncio.Queue()
            producer = asyncio.create_task(
                self._produce_sentences(text, language, sentences, timing), name="llm-stream"
            )
            self._reply_count += 1
            utterance_id = f"u{self._reply_count:04d}"
            sink = self._open_sink()
            seq = 0
            muted = False
            backchannel_word = ""
            try:
                # 相槌: 聞き取りの確定直後、LLM の 1 文目を待たずに鳴らす（音声モードだけ）
                if timing is not None and self._backchannels is not None:
                    clip = await self._pick_backchannel(text)
                    if clip is not None:
                        self._gate.mute()
                        muted = True
                        timing.t_backchannel = timing.t_audio_start = time.monotonic()
                        remaining = await sink.push(
                            SpokenSentence(utterance_id, seq, clip.text, clip.wav, clip.query))
                        seq += 1
                        self._gate.mute(expected_sec=remaining)
                        self._conversation.note_backchannel()
                        backchannel_word = clip.text
                while (sentence := await sentences.get()) is not None:
                    if backchannel_word:  # 「はい、…」を相槌と重ねない
                        sentence = strip_leading_backchannel(sentence, backchannel_word)
                        backchannel_word = ""
                        if not sentence:
                            continue
                    shown = self._conversation.redact(sentence)  # 読み上げは本物、画面とコンソールは伏せる
                    print(f"[井戸端AI] {shown}")
                    try:
                        wav, query = await self._tts.synthesize_with_query(sentence)
                    except httpx.HTTPError:
                        logger.warning("VOICEVOX 合成に失敗。この文は音声なしで続行します。")
                        continue
                    if not muted:
                        self._gate.mute()
                        muted = True
                        if timing is not None:
                            timing.t_audio_start = time.monotonic()
                    try:
                        remaining = await sink.push(
                            SpokenSentence(utterance_id, seq, shown, wav, query)
                        )
                    except Exception:
                        logger.warning("音声を流せません。この文はスキップします。", exc_info=True)
                        continue
                    seq += 1
                    # 安全装置: 再生終了の見込み + マージンで、通知が無くてもゲートは開く
                    self._gate.mute(expected_sec=remaining)
                await producer
                if seq > 0:
                    await sink.finish()
            finally:
                if not producer.done():
                    producer.cancel()
                if muted:
                    self._gate.schedule_unmute()
                elif self._edge is not None:
                    await self._send_state("idle")  # 1 文も流せなかった
            await self._show_sources()
            if timing is not None:
                timing.search_ms = self._dialogue.last_search_ms
                print(f"[計測] 発話終了→ {timing.summary()}")
            elif self._dialogue.last_search_ms is not None:
                print(f"[検索] {self._dialogue.last_search_ms:.0f}ms、"
                      f"参考資料 {len(self._dialogue.last_references)} 件")

    async def _produce_sentences(
        self,
        text: str,
        language: str,
        out: asyncio.Queue[str | None],
        timing: TurnTiming | None,
    ) -> None:
        """LLM の応答を文に切って out に積む。最後に None を積む。"""
        try:
            async for sentence in self._conversation.respond(text, language):
                if timing is not None and not timing.t_first_sentence:
                    timing.t_first_sentence = time.monotonic()
                await out.put(sentence)
        finally:
            if timing is not None:
                timing.t_llm_done = time.monotonic()
            await out.put(None)

    async def _show_sources(self) -> None:
        """応答が参考資料を使っていたら、出典のタイトルを出す（読み上げはしない）。"""
        titles = list(dict.fromkeys(
            f"{r.label}「{r.title}」" if r.label else r.title for r in self._dialogue.last_cited
        ))
        if not titles:
            return
        print(f"[出典] {' / '.join(titles)}")
        if self._edge is not None:
            await self._edge.send_json({"type": "sources", "titles": titles})

    async def _pick_backchannel(self, text: str) -> BackchannelClip | None:
        assert self._backchannels is not None
        if self._conversation.phase != "listen":
            return None  # 提案への返事（「うん」など）には相槌を打たない
        chosen = self._backchannels.choose(text, greeting_for())
        if chosen is None:
            return None
        return await self._backchannels.get(self._tts, *chosen)

    async def _make_lookup_query(self, text: str) -> str | None:
        """調べるための検索語（LLM が作る。本人の事情・氏名は含めない。ログに出さない）。"""
        try:
            raw = await asyncio.to_thread(
                self._llm.chat, config.llm.chat_model,
                [{"role": "system", "content": config.lookup.query_prompt},
                 {"role": "user", "content": text}],
                max_tokens=config.lookup.query_max_tokens, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            logger.warning("検索語を作れませんでした", exc_info=True)
            return None
        return self._dialogue.safe_query(raw)

    def _edge_id(self) -> str:
        return self._edge.edge_id if self._edge is not None else ""

    def _open_sink(self) -> SpeechSink:
        """1 応答ぶんの出力先を作る。"""
        if self._output_wav_dir is not None:
            return WavDirWriter(self._output_wav_dir / f"reply_{self._reply_count:02d}.wav")
        if self._edge is not None:
            return EdgeSpeaker(self._edge, config.edge.playback_timeout_margin_sec)
        return LocalSpeaker(self._tts.open_player())

    # ------------------------------------------------------------------
    # エッジ
    # ------------------------------------------------------------------

    def _on_edge_hello(self, hello: dict[str, Any]) -> None:
        """エッジがつながったら、拠点と季節（背景）と現在の状態を送る。"""
        async def greet() -> None:
            assert self._edge is not None
            await self._edge.send_theme(config.site, current_season())
            await self._edge.send_state("idle")

        task = asyncio.create_task(greet(), name="edge-hello")
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _spawn(self, coro: Coroutine[Any, Any, None], name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _watch_knowledge(self) -> None:
        """知識パックの差し替え（毎朝の作り直し）を、再起動なしで取り込む。"""
        while True:
            await asyncio.sleep(config.knowledge.reload_check_sec)
            await asyncio.to_thread(self._knowledge.reload_if_changed)

    def _shown_user_text(self, text: str) -> str:
        """字幕・コンソールに出す来訪者の言葉（氏名・地区・電話番号の答えは「お名前：伺いました」）。"""
        label = self._conversation.private_label
        return f"{label}：伺いました" if label else self._conversation.redact(text)

    async def _send_state(self, value: str) -> None:
        if self._edge is not None:
            await self._edge.send_state(value)

    async def _send_caption(self, text: str, *, role: str) -> None:
        if self._edge is not None:
            await self._edge.send_caption(text, final=True, role=role)

    async def _shutdown(self) -> None:
        await self._drain_turns()
        await self._conversation.end_session()
        await close_browser()
        if self._asr is not None:
            await self._asr.close()
        await self._tts.close()
        if self._edge_server is not None:
            await self._edge_server.stop()
        if self._edge is not None:
            await self._edge.stop()


def current_season(month: int | None = None) -> str:
    """月から季節（エッジの背景の切替に使う）を返す。"""
    m = month if month is not None else time.localtime().tm_mon
    return config.edge.seasons[(m - 3) % 12 // 3]


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
    parser = argparse.ArgumentParser(description="井戸端AI 音声往復 CLI（WP01）")
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
        "--audio-device", type=parse_device, default=None,
        help="音声入力デバイス（番号か名前の一部。省略時は .env の IDOBATA_AUDIO_INPUT_DEVICE）",
    )
    parser.add_argument(
        "--output-device", type=parse_device, default=None,
        help="TTS 音声出力デバイス（番号か名前の一部。省略時は .env の IDOBATA_AUDIO_OUTPUT_DEVICE）",
    )
    parser.add_argument(
        "--edge", action="store_true",
        help="マイク・スピーカー・アバターをブラウザ（エッジ）側に置く（http://localhost:8080/edge/）",
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

    if args.edge and args.output_wav_dir is not None:
        parser.error("--edge と --output-wav-dir は同時に指定できません")
    if args.input_wav and args.mode != "voice":
        parser.error("--input-wav は --mode voice と一緒に指定してください")

    app = VoiceLoop(
        mode=args.mode,
        input_wavs=args.input_wav,
        audio_device=args.audio_device,
        output_device=args.output_device,
        output_wav_dir=args.output_wav_dir,
        edge=args.edge,
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
