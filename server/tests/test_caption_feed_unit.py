"""caption_feed.py のユニットテスト。

CaptionFeed の final/partial 配信、is_recording 中の蓄積、
スロットリング、レイテンシ統計を検証する。
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.caption_feed import CaptionFeed
from src.schemas import TranscriptChunk


def _make_caption_feed() -> tuple[CaptionFeed, AsyncMock, MagicMock]:
    """テスト用 CaptionFeed とモックを生成する。"""
    bridge = AsyncMock()
    manager = MagicMock()
    manager.is_recording = True
    manager.suppress_captions = False
    manager.chat_active = False
    manager.accumulate_transcript = AsyncMock(return_value=None)

    feed = CaptionFeed(bridge=bridge, session_manager=manager)
    return feed, bridge, manager


def _make_chunk(
    text: str = "テスト",
    is_final: bool = True,
    latency: float = 1.0,
) -> TranscriptChunk:
    """テスト用 TranscriptChunk を生成する。"""
    now = time.monotonic()
    return TranscriptChunk(
        text=text,
        is_final=is_final,
        t_speech_start=now - latency,
        t_emitted=now,
    )


class TestFinalDispatch:
    """final チャンクの配信を検証する。"""

    async def test_final_sends_to_bridge(self) -> None:
        feed, bridge, _ = _make_caption_feed()
        chunk = _make_chunk("確定テキスト", is_final=True)

        await feed.dispatch(chunk)

        bridge.send_caption.assert_called_once_with("確定テキスト", final=True)

    async def test_final_accumulates_during_presenting(self) -> None:
        feed, _, manager = _make_caption_feed()
        manager.is_recording = True
        chunk = _make_chunk("確定テキスト", is_final=True)

        await feed.dispatch(chunk)

        manager.accumulate_transcript.assert_called_once_with("確定テキスト")

    async def test_final_does_not_accumulate_outside_presenting(self) -> None:
        feed, _, manager = _make_caption_feed()
        manager.is_recording = False
        chunk = _make_chunk("確定テキスト", is_final=True)

        await feed.dispatch(chunk)

        manager.accumulate_transcript.assert_not_called()

    async def test_final_records_in_history(self) -> None:
        feed, _, _ = _make_caption_feed()
        chunk = _make_chunk("履歴テスト", is_final=True)

        await feed.dispatch(chunk)

        recent = feed.get_recent_captions(5)
        assert "履歴テスト" in recent

    async def test_final_records_latency(self) -> None:
        feed, _, _ = _make_caption_feed()
        chunk = _make_chunk("テスト", is_final=True, latency=2.0)

        await feed.dispatch(chunk)

        stats = feed.get_latency_stats()
        assert stats["count"] == 1
        assert stats["avg"] > 1.5  # 約2秒のレイテンシ
        assert stats["max"] > 1.5


class TestPartialDispatch:
    """partial チャンクのスロットリングを検証する。"""

    async def test_partial_sends_to_bridge(self) -> None:
        feed, bridge, _ = _make_caption_feed()
        chunk = _make_chunk("中間テキスト", is_final=False)

        await feed.dispatch(chunk)

        bridge.send_caption.assert_called_once_with("中間テキスト", final=False)

    async def test_partial_throttling(self) -> None:
        feed, bridge, _ = _make_caption_feed()

        # 1回目: 送信される
        chunk1 = _make_chunk("partial1", is_final=False)
        await feed.dispatch(chunk1)

        # 直後の2回目: スロットリングで送信されない
        chunk2 = _make_chunk("partial2", is_final=False)
        await feed.dispatch(chunk2)

        # send_caption は1回のみ呼ばれている
        assert bridge.send_caption.call_count == 1

    async def test_partial_does_not_accumulate(self) -> None:
        feed, _, manager = _make_caption_feed()
        chunk = _make_chunk("中間テキスト", is_final=False)

        await feed.dispatch(chunk)

        manager.accumulate_transcript.assert_not_called()

    async def test_partial_after_throttle_window(self) -> None:
        feed, bridge, _ = _make_caption_feed()

        # 1回目
        chunk1 = _make_chunk("partial1", is_final=False)
        await feed.dispatch(chunk1)

        # スロットリング窓を過ぎたことをシミュレート
        with patch("src.caption_feed.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 0.3
            chunk2 = _make_chunk("partial2", is_final=False)
            await feed.dispatch(chunk2)

        # 2回目も送信される
        assert bridge.send_caption.call_count == 2


class TestLatencyStats:
    """レイテンシ統計の正確性を検証する。"""

    async def test_empty_stats(self) -> None:
        feed, _, _ = _make_caption_feed()
        stats = feed.get_latency_stats()
        assert stats == {"avg": 0.0, "max": 0.0, "count": 0}

    async def test_stats_with_multiple_finals(self) -> None:
        feed, _, _ = _make_caption_feed()

        await feed.dispatch(_make_chunk("a", latency=1.0))
        await feed.dispatch(_make_chunk("b", latency=3.0))
        await feed.dispatch(_make_chunk("c", latency=2.0))

        stats = feed.get_latency_stats()
        assert stats["count"] == 3
        assert stats["avg"] == pytest.approx(2.0, abs=0.2)
        assert stats["max"] == pytest.approx(3.0, abs=0.2)


class TestRecentCaptions:
    """直近字幕の取得を検証する。"""

    async def test_get_recent_captions(self) -> None:
        feed, _, _ = _make_caption_feed()

        for i in range(10):
            await feed.dispatch(_make_chunk(f"字幕{i}", is_final=True))

        recent = feed.get_recent_captions(3)
        assert len(recent) == 3
        assert recent[-1] == "字幕9"

    async def test_get_recent_captions_fewer_than_requested(self) -> None:
        feed, _, _ = _make_caption_feed()
        await feed.dispatch(_make_chunk("唯一の字幕", is_final=True))

        recent = feed.get_recent_captions(5)
        assert len(recent) == 1
        assert recent[0] == "唯一の字幕"
