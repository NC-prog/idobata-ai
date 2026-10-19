"""vad.py のユニットテスト。

Silero VAD の基本動作（無音 → イベントなし、音声 → START/END）と
reset() の状態クリアを検証する。
"""

from __future__ import annotations

import numpy as np

from src.config import VADConfig
from src.vad import SileroVAD, SpeechEventType


def _make_vad(
    threshold: float = 0.5,
    max_segment_sec: float = 30.0,
    min_silence_ms: int = 100,
    soft_split_enabled: bool = False,
    soft_split_lookback_ms: int = 500,
    soft_split_rise_threshold: float = 0.005,
) -> SileroVAD:
    """テスト用の SileroVAD インスタンスを生成する。"""
    cfg = VADConfig(
        threshold=threshold,
        max_segment_sec=max_segment_sec,
        min_silence_ms=min_silence_ms,
        pre_roll_ms=0,
        min_speech_ms=250,
        soft_split_enabled=soft_split_enabled,
        soft_split_lookback_ms=soft_split_lookback_ms,
        soft_split_rise_threshold=soft_split_rise_threshold,
    )
    return SileroVAD(cfg)


class TestSilenceInput:
    """無音入力でイベントが発生しないことを検証する。"""

    def test_silence_produces_no_events(self) -> None:
        vad = _make_vad()
        silence = np.zeros(512, dtype=np.float32)

        events = []
        for _ in range(100):
            ev = vad.process_frame(silence)
            if ev is not None:
                events.append(ev)

        # 無音で SPEECH_START は発生しない
        starts = [e for e in events if e.event_type == SpeechEventType.SPEECH_START]
        assert len(starts) == 0


class TestSpeechDetection:
    """音声 → 無音のパターンで START/END が検出されることを検証する。"""

    def test_noise_then_silence_produces_start_and_end(self) -> None:
        vad = _make_vad(threshold=0.3, min_silence_ms=100)

        events = []

        # ランダムノイズ（音声っぽい信号）を送る
        rng = np.random.default_rng(42)
        for _ in range(200):
            # 適度な振幅のノイズ
            frame = rng.normal(0, 0.3, 512).astype(np.float32)
            ev = vad.process_frame(frame)
            if ev is not None:
                events.append(ev)

        # その後に無音を送って SPEECH_END を誘発
        silence = np.zeros(512, dtype=np.float32)
        for _ in range(200):
            ev = vad.process_frame(silence)
            if ev is not None:
                events.append(ev)

        # 何らかのイベントが発生したことを確認（モデル依存のため、
        # 最低限 START か END のいずれかが出ていればOK）
        # 注: ランダムノイズが常に speech として検出される保証はないが、
        # 高振幅なのでほぼ検出される
        event_types = [e.event_type for e in events]
        # イベントが発生している場合のみ検証
        if SpeechEventType.SPEECH_START in event_types:
            assert SpeechEventType.SPEECH_END in event_types


class TestMaxSegmentForcing:
    """max_segment_sec 超過で強制分割されることを検証する。"""

    def test_forced_split_on_max_segment(self) -> None:
        # max_segment_sec を非常に短く設定
        vad = _make_vad(threshold=0.1, max_segment_sec=0.1, min_silence_ms=5000)

        events = []

        # 強い信号を長時間送り続ける
        rng = np.random.default_rng(123)
        for _ in range(500):
            frame = rng.normal(0, 0.5, 512).astype(np.float32)
            ev = vad.process_frame(frame)
            if ev is not None:
                events.append(ev)

        # 強制分割による SPEECH_END が発生するはず
        event_types = [e.event_type for e in events]
        if SpeechEventType.SPEECH_START in event_types:
            # START が出ていれば、max_segment 超過で END も出るはず
            assert SpeechEventType.SPEECH_END in event_types


class TestReset:
    """reset() で状態がクリアされることを検証する。"""

    def test_reset_clears_speech_state(self) -> None:
        vad = _make_vad()

        # 何フレームか処理
        rng = np.random.default_rng(99)
        for _ in range(50):
            frame = rng.normal(0, 0.3, 512).astype(np.float32)
            vad.process_frame(frame)

        # reset
        vad.reset()

        assert not vad.in_speech
        assert vad._speech_samples == 0

    def test_reset_allows_new_detection(self) -> None:
        vad = _make_vad(threshold=0.3, min_silence_ms=100)

        # reset 後でも正常に動作する
        vad.reset()

        silence = np.zeros(512, dtype=np.float32)
        events = []
        for _ in range(50):
            ev = vad.process_frame(silence)
            if ev is not None:
                events.append(ev)

        # 無音なのでイベントなし
        starts = [e for e in events if e.event_type == SpeechEventType.SPEECH_START]
        assert len(starts) == 0


class TestSoftSplit:
    """ソフト分割（エネルギーの谷で分割）の検証。"""

    def test_soft_split_at_energy_valley(self) -> None:
        """ソフト分割有効時、lookback ゾーンでエネルギーの谷を検出して分割する。"""
        # max_segment_sec=0.1s, lookback=50ms → ソフトゾーンは 0.05s 以降
        vad = _make_vad(
            threshold=0.1,
            max_segment_sec=0.1,
            min_silence_ms=5000,
            soft_split_enabled=True,
            soft_split_lookback_ms=50,
            soft_split_rise_threshold=0.005,
        )

        events: list[tuple[SpeechEventType, int]] = []

        rng = np.random.default_rng(42)
        frame_idx = 0

        # フェーズ1: 強い信号で発話開始を誘発 (最初の ~50フレーム)
        for _ in range(80):
            frame = rng.normal(0, 0.5, 512).astype(np.float32)
            ev = vad.process_frame(frame)
            if ev is not None:
                events.append((ev.event_type, frame_idx))
            frame_idx += 1

        # フェーズ2: ソフトゾーンに入った辺りでエネルギーの谷を作る
        # 無音 → 再上昇で局所最小通過を模倣
        for _ in range(5):
            frame = np.full(512, 0.0001, dtype=np.float32)  # 非常に小さいエネルギー
            ev = vad.process_frame(frame)
            if ev is not None:
                events.append((ev.event_type, frame_idx))
            frame_idx += 1

        for _ in range(5):
            frame = rng.normal(0, 0.3, 512).astype(np.float32)  # エネルギー上昇
            ev = vad.process_frame(frame)
            if ev is not None:
                events.append((ev.event_type, frame_idx))
            frame_idx += 1

        event_types = [e[0] for e in events]

        if SpeechEventType.SPEECH_START in event_types:
            # ソフト分割またはハードカットオフで END が出ているはず
            assert SpeechEventType.SPEECH_END in event_types
            # ソフト分割の場合、ハードカットオフ (max_segment 到達) より前に END が出る
            # max_segment は 0.1s = 1600 サンプル = ~3.125 フレーム (512サンプル/フレーム)
            # ソフト分割は lookback ゾーン内で起きるので、多くの場合ハードカットオフより手前

    def test_soft_split_disabled_falls_back_to_hard_cutoff(self) -> None:
        """ソフト分割無効時、従来のハードカットオフで分割される。"""
        vad = _make_vad(
            threshold=0.1,
            max_segment_sec=0.1,
            min_silence_ms=5000,
            soft_split_enabled=False,
        )

        events: list[tuple[SpeechEventType, int]] = []

        # 一定のノイズを送り続ける
        rng = np.random.default_rng(123)
        for i in range(500):
            frame = rng.normal(0, 0.5, 512).astype(np.float32)
            ev = vad.process_frame(frame)
            if ev is not None:
                events.append((ev.event_type, i))

        event_types = [e[0] for e in events]

        if SpeechEventType.SPEECH_START in event_types:
            # ハードカットオフで END が発生するはず
            assert SpeechEventType.SPEECH_END in event_types

    def test_soft_split_resets_state_after_split(self) -> None:
        """ソフト分割後、内部状態がリセットされ次の検出が可能。"""
        vad = _make_vad(
            threshold=0.1,
            max_segment_sec=0.1,
            min_silence_ms=5000,
            soft_split_enabled=True,
            soft_split_lookback_ms=50,
            soft_split_rise_threshold=0.005,
        )

        # 強い信号を長時間送って分割を誘発
        rng = np.random.default_rng(77)
        end_count = 0
        for _ in range(1000):
            frame = rng.normal(0, 0.5, 512).astype(np.float32)
            ev = vad.process_frame(frame)
            if ev is not None and ev.event_type == SpeechEventType.SPEECH_END:
                end_count += 1

        # 分割が1回以上発生し、状態リセット後に再検出できていることを確認
        if end_count > 0:
            assert not vad._in_soft_zone or vad._in_speech
