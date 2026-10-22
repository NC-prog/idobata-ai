"""WP02 A: カメラと「輪」（正面寄りの判定・滞留と離脱・人数・聞く/聞かないの制御・別れと区切り）。"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.camera import CameraClient, PartySnapshot, PartyState, PartyTracker, PersonObs, is_facing
from src.config import CameraConfig, PartyConfig
from src.main import VoiceLoop
from src.transcripts import TranscriptStore

CFG = PartyConfig()


def _kpts(nose: float, leye: float, reye: float, lear: float, rear: float,
          conf: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9, 0.9)) -> tuple[list[list[float]], list[float]]:
    xy = [[nose, 50], [leye, 45], [reye, 45], [lear, 47], [rear, 47]] + [[0, 0]] * 12
    return xy, list(conf) + [0.0] * 12


def test_is_facing_uses_ears_or_eyes() -> None:
    assert is_facing(*_kpts(160, 166, 154, 172, 148), 320, 0.5, 0.04)  # 耳の中点が鼻と同じ
    assert not is_facing(*_kpts(160, 166, 154, 150, 128), 320, 0.5, 0.04)  # 横向き（耳の中点が 21px ずれ）
    # 片方の耳が見えなければ目の中点で見る
    assert is_facing(*_kpts(160, 166, 154, 172, 0, conf=(0.9, 0.9, 0.9, 0.9, 0.1)), 320, 0.5, 0.04)
    # 鼻か目が見えなければ正面ではない（後ろ向き）
    assert not is_facing(*_kpts(160, 166, 154, 172, 148, conf=(0.2, 0.9, 0.9, 0.9, 0.9)), 320, 0.5, 0.04)


def _p(tid: int, ratio: float = 0.5, facing: bool = True) -> PersonObs:
    return PersonObs(tid, ratio, facing)


def test_party_after_dwell_and_leave_after_turning_away() -> None:
    t = PartyTracker(CFG)
    assert t.update([], 0.0).state is PartyState.NONE
    assert t.update([_p(1, ratio=0.2)], 0.2).state is PartyState.NONE  # 遠い人は数えない
    assert t.update([_p(1)], 1.0).state is PartyState.PRESENT
    assert t.update([_p(1)], 2.9).state is PartyState.PRESENT
    snap = t.update([_p(1)], 3.0)
    assert snap == PartySnapshot(PartyState.PARTY, 1, True)  # 正面寄りが 2 秒続いた
    assert t.update([_p(1, facing=False)], 4.0).state is PartyState.PARTY  # 横を向いても 3 秒は輪のまま
    assert t.update([_p(1, facing=False)], 5.9).state is PartyState.PARTY
    assert t.update([_p(1, facing=False)], 6.0).state is PartyState.PRESENT  # 3 秒で外れる
    assert t.update([], 7.0).state is PartyState.NONE


def test_two_people_count_and_max_three() -> None:
    t = PartyTracker(CFG)
    for now in (0.0, 1.0, 2.0):
        t.update([_p(1), _p(2)], now)
    assert t.update([_p(1), _p(2)], 2.1) == PartySnapshot(PartyState.PARTY, 2, True)
    for now in (3.0, 4.0, 5.0, 6.0):
        snap = t.update([_p(i) for i in range(1, 6)], now)
    assert snap.count == 3  # 輪は最大 3 人
    t.reset()
    assert t.snapshot.state is PartyState.NONE


def test_unseen_member_leaves_after_leave_sec() -> None:
    t = PartyTracker(CFG)
    for now in (0.0, 2.0):
        t.update([_p(7)], now)
    assert t.update([], 3.0).state is PartyState.PARTY  # 一瞬見えなくても（トラックが切れても）すぐには外さない
    assert t.update([], 5.0).state is PartyState.NONE


async def test_camera_client_stale_means_listening() -> None:
    cam = CameraClient(CameraConfig(stale_sec=0.2), CFG, Path("unused"))
    assert not cam.active and cam.listening  # まだフレームが来ていない＝従来どおり聞く
    cam.push_jpeg(b"\xff\xd8")
    assert cam.active and not cam.listening  # フレームが来ていて輪が無い＝聞かない
    await asyncio.sleep(0.25)
    assert not cam.active and cam.listening  # 止まったら耳を塞がない


class _Edge:
    edge_id = "edge-test"

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.states: list[str] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def send_state(self, value: str) -> None:
        self.states.append(value)

    async def send_caption(self, text: str, final: bool = True, role: str = "assistant") -> None:
        pass


def _loop() -> tuple[VoiceLoop, _Edge]:
    app = VoiceLoop(mode="voice", input_wavs=[], audio_device=None, output_device=None, output_wav_dir=None,
                    edge=True)
    edge = _Edge()
    app._edge = edge  # type: ignore[assignment]
    app._transcripts = TranscriptStore(Path(tempfile.mkdtemp(prefix="idobata-test-")), "etajima", 7)
    return app, edge


async def test_segment_is_not_transcribed_without_party() -> None:
    app, _edge = _loop()
    assert app._camera is not None
    calls: list[int] = []

    class _ASR:
        async def transcribe_segment(self, *a: Any, **kw: Any) -> Any:
            calls.append(1)
            if False:
                yield None

    app._asr = _ASR()  # type: ignore[assignment]
    app._camera.push_jpeg(b"\xff\xd8")  # カメラは動いている、輪は無い
    app._party_at_start = app._camera_listening()
    await app._on_segment([np.zeros(512, dtype=np.float32)], 0.0)
    assert calls == []  # ASR に渡さない
    app._camera.tracker.snapshot = PartySnapshot(PartyState.PARTY, 1, True)
    await app._on_segment([np.zeros(512, dtype=np.float32)], 0.0)
    assert calls == [1]
    await app._tts.close()


async def test_farewell_once_and_bye_ends_conversation(monkeypatch: Any) -> None:
    from dataclasses import replace

    import src.main as main_mod

    monkeypatch.setattr(main_mod, "config", replace(main_mod.config, party=PartyConfig(bye_sec=0.2)))
    app, edge = _loop()
    assert app._camera is not None
    said: list[str] = []

    async def fake_respond(text: str, language: str, timing: object, source: Any = None,
                           trigger: str | None = None) -> None:
        said.extend([s async for s in source()])

    app._respond_and_speak = fake_respond  # type: ignore[method-assign]
    app._conversation._ensure_session("ja")
    assert app._conversation._session is not None
    app._conversation._session.turns = 2  # 会話の途中
    app._last_utterance_at = time.monotonic()
    party, none = PartySnapshot(PartyState.PARTY, 1, True), PartySnapshot(PartyState.NONE, 0, False)
    app._camera.tracker.snapshot = none
    await app._on_party_change(party, none)
    await app._on_party_change(party, none)
    await asyncio.sleep(0.4)
    assert said == ["お気をつけて。またどうぞ。"]  # 1 会話 1 回
    assert edge.sent[0] == {"type": "party", "state": "NONE", "count": 0, "facing": False}
    assert app._conversation.session_language is None and "idle" in edge.states  # 8 秒（ここでは 0.2 秒）で区切る
    await app._tts.close()


def test_simple_tracker_keeps_ids_by_iou_and_center() -> None:
    from src.camera import SimpleTracker

    t = SimpleTracker(CameraConfig(track_max_missed=2))
    a, b = np.array([10, 10, 60, 170.0]), np.array([200, 20, 260, 175.0])
    first = t.update(np.stack([a, b]), 320, 180)
    assert first == [1, 2]
    # 少し動いても同じ ID（IoU）。並び順が入れ替わっても対応付ける
    assert t.update(np.stack([b + 4, a + 3]), 320, 180) == [2, 1]
    # 速く動いて重ならなくても、中心が近ければ同じ人
    assert t.update(np.stack([a + 40, b + 4]), 320, 180) == [1, 2]
    # 見えなくなって track_max_missed を超えたら忘れる → 次は新しい ID
    for _ in range(3):
        t.update(np.zeros((0, 4)), 320, 180)
    assert t.update(np.stack([a]), 320, 180) == [3]
    t.reset()
    assert t.update(np.stack([a]), 320, 180) == [1]


def test_pose_detector_on_local_weights() -> None:
    """重みが $DATA/models/rtmpose にあれば、本物の推論（CPU）が動き、誰もいない画面では人を出さない（無ければ飛ばす）。"""
    import pytest

    from src.camera import PoseDetector
    from src.config import config

    root = config.data_dir / "models" / "rtmpose"
    if not root.is_dir():
        pytest.skip("姿勢推定の重みが無い")
    import cv2

    img = np.full((180, 320, 3), 40, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    det = PoseDetector(CameraConfig(device="cpu"), CFG, root)
    out = det.detect(buf.tobytes(), debug=True)
    assert out is not None and out.people == [] and out.debug_jpeg  # 誰もいない画面


async def test_missing_weights_means_listening(tmp_path: Path) -> None:
    cam = CameraClient(CameraConfig(), CFG, tmp_path / "none")
    task = asyncio.create_task(cam.run())
    cam.push_jpeg(b"\xff\xd8")
    await asyncio.wait_for(task, 10)  # 読めなければ推論ループを抜ける
    assert cam.failed and not cam.active and cam.listening  # フレームが来ていても耳を塞がない
