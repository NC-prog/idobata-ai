"""カメラと「輪」（WP02 A、camera_client）。

エッジ（ブラウザ）が撮ったカメラの映像を、320×180 の JPEG・5fps で WebSocket に載せて送ってくる
（カメラを Python から直接開く経路は作らない。別機のエッジでも同じ形で動かすため）。
ここではそれを **メモリ上でだけ** YOLO11-pose に掛け、目の前の「輪」（会話の相手の 1〜3 人）を決める。
映像フレームはディスクに書かない。顔識別（個人特定）はしない。人の対応付けはトラック ID（セッション限り）。

輪の判定（人ごと、毎フレーム）:
- 近い: 矩形の高さが画面の高さの party.min_box_ratio 以上
- 正面寄り: 鼻・両目が見えていて、左右の耳（無ければ目）の中点が鼻から画面幅の party.face_center_tol 以内
- 近くて正面寄りが party.dwell_sec 続いたら輪の中、外れて party.leave_sec で輪の外（最大 party.max_members 人）

状態: NONE（誰もいない）→ PRESENT（近い人はいるが輪は無い）→ PARTY（輪に 1 人以上）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .config import CameraConfig, PartyConfig

logger = logging.getLogger(__name__)

# COCO の 17 キーポイントのうち顔の 5 つ
_NOSE, _LEYE, _REYE, _LEAR, _REAR = 0, 1, 2, 3, 4


class PartyState(str, Enum):
    NONE = "NONE"
    PRESENT = "PRESENT"
    PARTY = "PARTY"


@dataclass(frozen=True)
class PersonObs:
    """1 フレームの 1 人（矩形の高さの割合と、正面寄りか）。"""

    track_id: int
    box_h_ratio: float
    facing: bool


@dataclass(frozen=True)
class PartySnapshot:
    state: PartyState
    count: int
    facing: bool


@dataclass
class _Track:
    first_ok: float | None = None
    last_ok: float | None = None
    last_seen: float = 0.0
    in_party: bool = False


def is_facing(kpts_xy: Sequence[Sequence[float]], kpts_conf: Sequence[float], frame_w: float,
              kpt_conf: float, tol: float) -> bool:
    """鼻・両目が見えていて、左右の耳（無ければ目）の中点が鼻の x から画面幅の tol 以内か。"""
    def seen(i: int) -> bool:
        return kpts_conf[i] >= kpt_conf

    if not (seen(_NOSE) and seen(_LEYE) and seen(_REYE)):
        return False
    pair = (_LEAR, _REAR) if seen(_LEAR) and seen(_REAR) else (_LEYE, _REYE)
    mid = (kpts_xy[pair[0]][0] + kpts_xy[pair[1]][0]) / 2
    return abs(mid - kpts_xy[_NOSE][0]) <= tol * frame_w


class PartyTracker:
    """フレームごとの人の観測から、輪（PARTY）とその人数を決める（推論とは切り離した規則だけ）。"""

    def __init__(self, cfg: PartyConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self.snapshot = PartySnapshot(PartyState.NONE, 0, False)

    def reset(self) -> None:
        self._tracks.clear()
        self.snapshot = PartySnapshot(PartyState.NONE, 0, False)

    def update(self, people: Sequence[PersonObs], now: float) -> PartySnapshot:
        cfg = self._cfg
        seen = {p.track_id: p for p in people}
        near_now = any(p.box_h_ratio >= cfg.min_box_ratio for p in people)
        for tid in seen:
            self._tracks.setdefault(tid, _Track()).last_seen = now
        members = sum(1 for t in self._tracks.values() if t.in_party)
        for tid, t in list(self._tracks.items()):
            obs = seen.get(tid)
            ok = obs is not None and obs.box_h_ratio >= cfg.min_box_ratio and obs.facing
            if ok:
                t.first_ok = t.first_ok if t.first_ok is not None else now
                t.last_ok = now
                if not t.in_party and now - t.first_ok >= cfg.dwell_sec and members < cfg.max_members:
                    t.in_party = True
                    members += 1
            else:
                t.first_ok = None
                if t.in_party and (t.last_ok is None or now - t.last_ok >= cfg.leave_sec):
                    t.in_party = False
                    members -= 1
            if not t.in_party and now - t.last_seen >= cfg.leave_sec:
                del self._tracks[tid]  # 見えなくなった人の記録は残さない
        in_party = [tid for tid, t in self._tracks.items() if t.in_party]
        if in_party:
            facing = any(seen[tid].facing for tid in in_party if tid in seen)
            self.snapshot = PartySnapshot(PartyState.PARTY, len(in_party), facing)
        elif near_now:
            self.snapshot = PartySnapshot(PartyState.PRESENT, 0, any(p.facing for p in people))
        else:
            self.snapshot = PartySnapshot(PartyState.NONE, 0, False)
        return self.snapshot


@dataclass
class Detection:
    people: list[PersonObs]
    infer_ms: float
    debug_jpeg: bytes | None = None


class PoseDetector:
    """YOLO11-pose（ultralytics）。JPEG を受け取り、人ごとの矩形・キーポイントから PersonObs を作る。

    ultralytics の利用状況の送信と、実行時の自動インストールは止める（外に何も送らない）。
    """

    def __init__(self, cfg: CameraConfig, party: PartyConfig, model_path: Path) -> None:
        self._cfg = cfg
        self._party = party
        os.environ.setdefault("YOLO_OFFLINE", "1")
        import ultralytics

        yolo: Any = ultralytics
        yolo.settings.update({"sync": False})
        self._model: Any = yolo.YOLO(str(model_path))
        self._quantize = "fp16" if cfg.device != "cpu" else None

    def detect(self, jpeg: bytes, debug: bool = False) -> Detection | None:
        import cv2

        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        results: Any = self._model.track(img, imgsz=self._cfg.imgsz, device=self._cfg.device, conf=self._cfg.det_conf,
                                    classes=[0], quantize=self._quantize, persist=True, tracker="bytetrack.yaml",
                                    verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000
        r = results[0]
        people: list[PersonObs] = []
        if r.boxes is not None and r.keypoints is not None and len(r.boxes):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-(i + 1) for i in range(len(r.boxes))]
            kxy: list[Any] = r.keypoints.xy.tolist()
            kconf: list[Any] = r.keypoints.conf.tolist() if r.keypoints.conf is not None else [[0.0] * 17] * len(ids)
            for tid, box, xy, conf in zip(ids, r.boxes.xyxy.tolist(), kxy, kconf, strict=True):
                people.append(PersonObs(int(tid), (box[3] - box[1]) / h,
                                        is_facing(xy, conf, w, self._cfg.kpt_conf, self._party.face_center_tol)))
        dbg = None
        if debug:
            ok, buf = cv2.imencode(".jpg", r.plot())
            dbg = buf.tobytes() if ok else None
        return Detection(people, infer_ms, dbg)


PartyHandler = Callable[[PartySnapshot, PartySnapshot], Awaitable[None]]


@dataclass
class CameraClient:
    """エッジから届いた最新のフレームだけを推論に掛け、輪が変わったら知らせる。"""

    cfg: CameraConfig
    party_cfg: PartyConfig
    model_path: Path
    on_change: PartyHandler | None = None
    debug: bool = False
    tracker: PartyTracker = field(init=False)
    debug_jpeg: bytes | None = field(default=None, init=False)
    _latest: bytes | None = field(default=None, init=False)
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _last_frame_at: float | None = field(default=None, init=False)
    _detector: PoseDetector | None = field(default=None, init=False)
    _times: deque[float] = field(default_factory=lambda: deque(maxlen=300), init=False)
    _frames: int = field(default=0, init=False)
    received: int = field(default=0, init=False)
    # 最後のフレームの人ごとの（矩形の高さの割合、正面寄りか）。距離のしきい値を測るときに /edge/debug で見る
    last_people: list[tuple[float, bool]] = field(default_factory=list, init=False)
    _started_at: float | None = field(default=None, init=False)
    _stale_warned: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.tracker = PartyTracker(self.party_cfg)

    def push_jpeg(self, data: bytes) -> None:
        """エッジからの JPEG（メモリに 1 枚だけ持つ。古いものは捨てる）。"""
        self._latest = data
        self._last_frame_at = time.monotonic()
        self.received += 1
        if self._started_at is None:
            self._started_at = self._last_frame_at
        self._event.set()

    @property
    def active(self) -> bool:
        """フレームが最近届いているか（届かなければ輪の判定はしない＝従来どおり聞く）。"""
        if self._last_frame_at is None:
            return False
        stale = time.monotonic() - self._last_frame_at > self.cfg.stale_sec
        if stale and not self._stale_warned:
            logger.warning("カメラのフレームが %.0f 秒届いていません。輪の判定をやめて、従来どおり聞きます",
                           self.cfg.stale_sec)
            self._stale_warned = True
        elif not stale:
            self._stale_warned = False
        return not stale

    @property
    def snapshot(self) -> PartySnapshot:
        return self.tracker.snapshot

    @property
    def listening(self) -> bool:
        """対話に流してよいか（輪がある、またはカメラが動いていない）。"""
        return not self.active or self.snapshot.state is PartyState.PARTY

    def timing_summary(self) -> str:
        if not self._times:
            return "推論なし"
        ts = sorted(self._times)
        span = (self._last_frame_at or 0) - (self._started_at or 0)
        rate = f"受信 {self.received / span:.1f}fps、" if span > 0 else ""
        return (f"{rate}推論 {len(ts)} フレーム: 中央値 {statistics.median(ts):.1f}ms / "
                f"95% {ts[int(len(ts) * 0.95) - 1 if len(ts) > 1 else 0]:.1f}ms")

    async def run(self) -> None:
        """推論ループ（モデルは最初のフレームが来てから読む）。"""
        while True:
            await self._event.wait()
            self._event.clear()
            data, self._latest = self._latest, None
            if data is None:
                continue
            if self._detector is None:
                self._detector = await asyncio.to_thread(PoseDetector, self.cfg, self.party_cfg, self.model_path)
                logger.info("YOLO11-pose を読み込みました（%s、imgsz %d）", self.model_path.name, self.cfg.imgsz)
            try:
                det = await asyncio.to_thread(self._detector.detect, data, self.debug)
            except Exception:
                logger.exception("姿勢推定に失敗しました（このフレームは使いません）")
                continue
            if det is None:
                continue
            self._times.append(det.infer_ms)
            self.last_people = [(round(p.box_h_ratio, 3), p.facing) for p in det.people]
            self._frames += 1
            if self._frames % self.cfg.timing_log_every == 0:
                logger.info("カメラ: %s", self.timing_summary())
            if self.debug:
                self.debug_jpeg = det.debug_jpeg
            before = self.tracker.snapshot
            after = self.tracker.update(det.people, time.monotonic())
            if after != before:
                logger.info("輪: %s → %s（%d 人、正面 %s）", before.state.value, after.state.value, after.count,
                            after.facing)
                if self.on_change is not None:
                    await self.on_change(before, after)
