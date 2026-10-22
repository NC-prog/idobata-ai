"""セッション限りの話者の区別（WP02 B）。

発話区間ごとに話者埋め込み（speechbrain の ECAPA-TDNN、Apache-2.0）を取り、会話の中だけでオンラインに
まとめる（コサイン類似度）。話者は「A」「B」「C」… と番号だけ。登録・照合はしない。
埋め込みとまとまりはメモリだけに持ち、会話が区切れたら捨てる（reset）。

受け付けの規則（A-3 の「輪が無いときは聞かない」と二重の網）:
- その話者が最初に現れたときに輪が PARTY だった → 相手
- 最初に現れたのが NONE（誰もいない）→ 捨てる（後ろの会話・職員の声）。その後に輪の中で話しても捨てる
- 最初に現れたのが PRESENT（近づいてくる途中）→ 保留。輪ができてから話せば相手にする
  （話しかけながら近づいてくる人を、ずっと聞かないことにしないため）
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .camera import PartyState
from .config import SpeakerConfig

logger = logging.getLogger(__name__)


@dataclass
class _Cluster:
    label: str
    total: np.ndarray  # 正規化した埋め込みの和（重心の向きだけ使う）
    n: int
    accepted: bool | None  # True 相手 / False 捨てる / None 保留（PRESENT で初めて現れた）

    @property
    def centroid(self) -> np.ndarray:
        v: np.ndarray = self.total / (np.linalg.norm(self.total) or 1.0)
        return v


@dataclass(frozen=True)
class SpeakerTurn:
    """1 区間の話者の判定。"""

    label: str
    accepted: bool
    new: bool
    similarity: float


@dataclass
class SpeakerClusters:
    """オンラインのまとまり（埋め込みの計算とは切り離した規則だけ）。"""

    cfg: SpeakerConfig
    _clusters: list[_Cluster] = field(default_factory=list)
    _next: int = 0

    def reset(self) -> None:
        if self._clusters:
            logger.info("話者の区別を捨てました（%d 人）", len(self._clusters))
        self._clusters.clear()
        self._next = 0

    @property
    def accepted_count(self) -> int:
        return sum(1 for c in self._clusters if c.accepted)

    def assign(self, emb: np.ndarray, voiced_sec: float, state: PartyState) -> SpeakerTurn:
        e = emb / (np.linalg.norm(emb) or 1.0)
        best, sim = None, -1.0
        for c in self._clusters:
            s = float(c.centroid @ e)
            if s > sim:
                best, sim = c, s
        short = voiced_sec < self.cfg.min_new_speaker_sec
        if best is not None and (sim >= self.cfg.same_threshold or short):
            # 同じ話者（短い区間は埋め込みが不安定なので、新しい話者を作らず近い話者に寄せ、重心も動かさない）
            if not short:
                best.total = best.total + e
                best.n += 1
            if best.accepted is None and state is PartyState.PARTY:
                best.accepted = True
                logger.info("話者 %s を相手にしました（輪ができてから話した）", best.label)
            return SpeakerTurn(best.label, bool(best.accepted), False, sim)
        accepted = {PartyState.PARTY: True, PartyState.NONE: False}.get(state)
        cluster = _Cluster(self._new_label(), e.copy(), 1, accepted)
        self._clusters.append(cluster)
        if len(self._clusters) > self.cfg.max_clusters:
            # 多すぎたら、相手でないもの（無ければいちばん古いもの）から捨てる
            drop = next((c for c in self._clusters if not c.accepted), self._clusters[0])
            self._clusters.remove(drop)
        logger.info("新しい話者 %s（%s、最も近い話者との類似度 %.2f）", cluster.label,
                    {True: "相手", False: "輪の外", None: "保留"}[accepted], sim)
        return SpeakerTurn(cluster.label, bool(accepted), True, sim)

    def _new_label(self) -> str:
        n, self._next = self._next, self._next + 1
        letters = string.ascii_uppercase
        return letters[n] if n < len(letters) else f"{letters[n // len(letters) - 1]}{letters[n % len(letters)]}"


class SpeakerEmbedder:
    """ECAPA-TDNN の埋め込み（重みは $DATA/models/。最初の区間で読む）。"""

    def __init__(self, cfg: SpeakerConfig, model_root: Path) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import-untyped]

        device = cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(cfg.cpu_threads)
        self._torch: Any = torch
        self._device = device
        self._model: Any = EncoderClassifier.from_hparams(
            source=cfg.model_source, savedir=str(model_root / cfg.model_dir), run_opts={"device": device})
        logger.info("話者埋め込み（%s）を読み込みました（%s）", cfg.model_source, device)

    def embed(self, wave: np.ndarray) -> np.ndarray:
        """16kHz の波形（float32）→ 正規化した埋め込み。"""
        with self._torch.no_grad():
            x = self._torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None].to(self._device)
            e: np.ndarray = self._model.encode_batch(x).squeeze().float().cpu().numpy()
        return e / (np.linalg.norm(e) or 1.0)
