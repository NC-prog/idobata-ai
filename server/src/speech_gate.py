"""自己発話ゲート。

TTS 再生中にマイクが拾った自己発話が字幕化されるのを防ぐ。
talk_start 〜 talk_end + 後マージンの間、VAD へのフレーム投入を停止する。

使い方:
- VoiceLoop._speak() が mute() / schedule_unmute() を呼ぶ
- VoiceLoop._run_pipeline() が is_muted を毎フレーム確認し、
  True の間はフレームを VAD に渡さない

安全装置: mute(expected_sec=...) で想定再生時間を渡すと、
「想定再生時間 + 後マージン + 安全マージン」後に自動で unmute する。
再生処理が戻らず schedule_unmute() が呼ばれなくても、ゲートが閉じたままにならない。
schedule_unmute() が先に呼ばれた場合はそちらが優先（安全装置は取り消す）。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# 後マージンのデフォルト値（秒）
DEFAULT_POST_MARGIN_SEC: float = 0.7
# 安全装置の追加マージンのデフォルト値（秒）
DEFAULT_SAFETY_MARGIN_SEC: float = 2.0


class SpeechGate:
    """TTS 再生窓で VAD パスをミュートするゲート。"""

    def __init__(
        self,
        post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
        safety_margin_sec: float = DEFAULT_SAFETY_MARGIN_SEC,
    ) -> None:
        self._muted = False
        self._post_margin = post_margin_sec
        self._safety_margin = safety_margin_sec
        self._unmute_task: asyncio.Task[None] | None = None
        self._safety_task: asyncio.Task[None] | None = None

    @property
    def is_muted(self) -> bool:
        """ゲートが閉じている（VAD フレーム投入停止中）かどうか。"""
        return self._muted

    def mute(self, expected_sec: float | None = None) -> None:
        """ゲートを閉じる。

        遅延 unmute がスケジュール済みならキャンセルする。
        expected_sec（想定再生時間）を渡すと安全装置の自動 unmute を（張り直して）予約する。
        閉じたまま呼び直してもよい（合成前に mute()、合成後に mute(expected_sec) など）。
        """
        if self._unmute_task is not None and not self._unmute_task.done():
            self._unmute_task.cancel()
            self._unmute_task = None
        if not self._muted:
            self._muted = True
            logger.info("[GATE] mute — TTS 再生開始、VAD フレーム投入停止")
        if expected_sec is not None:
            self._cancel_safety()
            delay = expected_sec + self._post_margin + self._safety_margin
            self._safety_task = asyncio.create_task(
                self._safety_unmute(delay), name="speech_gate_safety_unmute",
            )

    def schedule_unmute(self) -> None:
        """後マージン経過後にゲートを開く（安全装置は取り消す）。"""
        self._cancel_safety()
        if self._unmute_task is not None and not self._unmute_task.done():
            self._unmute_task.cancel()
        self._unmute_task = asyncio.create_task(
            self._delayed_unmute(), name="speech_gate_unmute",
        )

    def _cancel_safety(self) -> None:
        if self._safety_task is not None and not self._safety_task.done():
            self._safety_task.cancel()
        self._safety_task = None

    async def _safety_unmute(self, delay: float) -> None:
        """再生が戻らない場合に備えた自動 unmute。"""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return  # schedule_unmute() / 新しい mute() が先に来た
        if self._muted:
            self._muted = False
            logger.warning(
                "[GATE] unmute（安全装置）— mute から %.1f秒経っても再生完了の通知がない。"
                "VAD フレーム投入を再開",
                delay,
            )

    async def _delayed_unmute(self) -> None:
        """後マージン待機後にミュートを解除する。"""
        try:
            await asyncio.sleep(self._post_margin)
            self._muted = False
            logger.info(
                "[GATE] unmute — 後マージン %.1f秒経過、VAD フレーム投入再開",
                self._post_margin,
            )
        except asyncio.CancelledError:
            pass  # 新しい mute() で上書きされた
