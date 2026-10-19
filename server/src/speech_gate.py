"""自己発話ゲート。

TTS 再生中にマイクが拾った自己発話が字幕化されるのを防ぐ。
talk_start 〜 talk_end + 後マージンの間、VAD へのフレーム投入を停止する。

使い方:
- SessionManager._announce() が mute() / schedule_unmute() を呼ぶ
- CLIApp._run_audio_pipeline() が is_muted を毎フレーム確認し、
  True の間はフレームを VAD に渡さない
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# 後マージンのデフォルト値（秒）
DEFAULT_POST_MARGIN_SEC: float = 0.7


class SpeechGate:
    """TTS 再生窓で VAD パスをミュートするゲート。"""

    def __init__(self, post_margin_sec: float = DEFAULT_POST_MARGIN_SEC) -> None:
        self._muted = False
        self._post_margin = post_margin_sec
        self._unmute_task: asyncio.Task[None] | None = None

    @property
    def is_muted(self) -> bool:
        """ゲートが閉じている（VAD フレーム投入停止中）かどうか。"""
        return self._muted

    def mute(self) -> None:
        """ゲートを閉じる。

        遅延 unmute がスケジュール済みならキャンセルする。
        """
        if self._unmute_task is not None and not self._unmute_task.done():
            self._unmute_task.cancel()
            self._unmute_task = None
        if not self._muted:
            self._muted = True
            logger.info("[GATE] mute — TTS 再生開始、VAD フレーム投入停止")

    def schedule_unmute(self) -> None:
        """後マージン経過後にゲートを開く。"""
        if self._unmute_task is not None and not self._unmute_task.done():
            self._unmute_task.cancel()
        self._unmute_task = asyncio.create_task(
            self._delayed_unmute(), name="speech_gate_unmute",
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
