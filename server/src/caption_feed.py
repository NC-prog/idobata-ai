"""字幕配信モジュール。

ASR からの TranscriptChunk を受け取り、Edge Bridge への字幕送信と
セッションへのテキスト蓄積を行う。partial 字幕はスロットリングする。

WP00: townhall の session_manager は持ち込まない（WP03 で対話FSMとして書き直す）ため、
セッションへの依存は CaptionSession プロトコルに置き換えている。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from .edge_bridge import EdgeBridge
from .schemas import TranscriptChunk

logger = logging.getLogger(__name__)

# Web パネルへの字幕通知コールバック型
# (text, is_final, is_recording) -> None
CaptionListener = Callable[[str, bool, bool], Coroutine[Any, Any, None]]

# partial 字幕の送信間隔（秒）
_PARTIAL_THROTTLE_SEC = 0.2


class CaptionSession(Protocol):
    """CaptionFeed が参照するセッション側のインターフェース。"""

    @property
    def chat_active(self) -> bool:
        """会話ループ中か（True なら final を handle_chat_utterance へ回す）。"""
        ...

    @property
    def suppress_captions(self) -> bool:
        """人声由来の字幕表示・蓄積を止めるか（townhall のクイズ状態相当）。"""
        ...

    @property
    def is_recording(self) -> bool:
        """確定字幕を蓄積する状態か（townhall の PRESENTING 相当）。"""
        ...

    async def handle_chat_utterance(self, text: str) -> str:
        """会話ループの 1 発話を処理する。"""
        ...

    async def accumulate_transcript(self, text: str) -> str | None:
        """確定字幕を蓄積する。"""
        ...


class CaptionFeed:
    """ASR 結果を Edge Bridge とセッションに配信する。"""

    def __init__(
        self,
        bridge: EdgeBridge,
        session_manager: CaptionSession,
        history_size: int = 50,
    ) -> None:
        self._bridge = bridge
        self._manager = session_manager
        self._history_size = history_size

        # 直近の final 字幕履歴
        self._recent_captions: deque[str] = deque(maxlen=history_size)

        # レイテンシ記録
        self._latencies: list[float] = []

        # partial スロットリング用
        self._last_partial_sent = 0.0

        # Web パネル用リスナー
        self._listeners: list[CaptionListener] = []

    def add_listener(self, listener: CaptionListener) -> None:
        """字幕リスナーを追加する。"""
        self._listeners.append(listener)

    async def _notify_listeners(self, text: str, is_final: bool, is_recording: bool) -> None:
        """登録済みリスナーに通知する。"""
        for listener in self._listeners:
            try:
                await listener(text, is_final, is_recording)
            except Exception:
                logger.exception("字幕リスナー通知中にエラー")

    async def dispatch(self, chunk: TranscriptChunk) -> None:
        """TranscriptChunk を処理する。

        - final: bridge に確定字幕を送信 + PRESENTING 中なら session_manager に蓄積
        - partial: 200ms スロットリングで bridge に中間字幕を送信
        """
        if chunk.is_final:
            await self._handle_final(chunk)
        else:
            await self._handle_partial(chunk)

    async def _handle_final(self, chunk: TranscriptChunk) -> None:
        """確定字幕を処理する。"""
        # CHAT 中は確定テキストを会話ループへ回す（通常の字幕表示・蓄積はしない）
        if self._manager.chat_active:
            logger.info("CHAT 発話として処理: %s", chunk.text[:40])
            await self._manager.handle_chat_utterance(chunk.text)
            return

        # クイズ状態中は人声由来の字幕表示・蓄積をスキップ（ASR自体は継続）
        if self._manager.suppress_captions:
            logger.debug("クイズ中のため人声字幕をスキップ: %s", chunk.text[:40])
            return

        # bridge に送信
        await self._bridge.send_caption(chunk.text, final=True)

        # 履歴に追加
        self._recent_captions.append(chunk.text)

        # レイテンシ記録
        latency = chunk.latency_sec
        self._latencies.append(latency)
        logger.info("字幕確定: %s (レイテンシ: %.2f秒)", chunk.text[:40], latency)

        # PRESENTING 中なら session_manager に蓄積（字幕送信なし）
        is_recording = self._manager.is_recording
        if is_recording:
            result = await self._manager.accumulate_transcript(chunk.text)
            if result:
                logger.info("ローリングサマリー更新: %s...", result[:60])

        # リスナーに通知
        await self._notify_listeners(chunk.text, is_final=True, is_recording=is_recording)

    async def _handle_partial(self, chunk: TranscriptChunk) -> None:
        """中間字幕をスロットリング付きで処理する。"""
        # CHAT 中は中間字幕を表示しない（会話ループは final のみ扱う）
        if self._manager.chat_active:
            return

        # クイズ状態中は人声由来の字幕をスキップ
        if self._manager.suppress_captions:
            return

        now = time.monotonic()
        if now - self._last_partial_sent < _PARTIAL_THROTTLE_SEC:
            return

        self._last_partial_sent = now
        await self._bridge.send_caption(chunk.text, final=False)

        # リスナーに通知
        is_recording = self._manager.is_recording
        await self._notify_listeners(chunk.text, is_final=False, is_recording=is_recording)

    def get_latency_stats(self) -> dict[str, float]:
        """レイテンシ統計を返す。"""
        if not self._latencies:
            return {"avg": 0.0, "max": 0.0, "count": 0}

        return {
            "avg": sum(self._latencies) / len(self._latencies),
            "max": max(self._latencies),
            "count": float(len(self._latencies)),
        }

    def get_recent_captions(self, n: int = 5) -> list[str]:
        """直近 N 件の確定字幕を返す。"""
        captions = list(self._recent_captions)
        return captions[-n:]
