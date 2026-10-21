"""相槌（WP04 A-1）。

聞き取りが確定した直後、LLM を呼ぶ前に短い相槌を 1 つ鳴らして、体感の待ちを消す。
候補は ``server/context/backchannel.yaml``。簡単な規則（打ち明け・質問・雑談・短い返事）で選び、
同じ区分の中では順番に使う（同じ相槌を続けない）。
起動時に VOICEVOX で事前合成し、末尾に無音（pause_sec）を足してキャッシュする。
"""

from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import yaml

from .timeofday import all_greetings, greeting_for

logger = logging.getLogger(__name__)


class Synthesizer(Protocol):
    async def synthesize_with_query(
        self, text: str, speaker_id: int | None = None
    ) -> tuple[bytes, dict[str, Any]]: ...


@dataclass
class _Category:
    name: str
    patterns: list[re.Pattern[str]]
    candidates: list[str]
    pause_sec: float
    turn: int = 0


@dataclass(frozen=True)
class BackchannelClip:
    """合成済みの相槌（末尾に無音を足した WAV と、口パク用の audio_query）。"""

    text: str
    wav: bytes
    query: dict[str, Any] = field(default_factory=dict)


class BackchannelBank:
    """相槌の選択と、事前合成のキャッシュ。"""

    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._pause = float(data.get("pause_sec", 0.3))
        self._none = [re.compile(p) for p in data.get("no_backchannel") or []]
        self._categories = [
            _Category(
                name=str(c["name"]),
                patterns=[re.compile(p) for p in c.get("patterns") or []],
                candidates=[str(x) for x in c.get("candidates") or []],
                pause_sec=float(c.get("pause_sec", self._pause)),
            )
            for c in data.get("categories") or []
        ]
        default = [str(x) for x in data.get("default") or []]
        self._default = _Category("default", [], default, self._pause)
        self._cache: dict[tuple[str, float], BackchannelClip] = {}

    def choose(self, text: str, greeting: str = "") -> tuple[str, float] | None:
        """相槌の語と無音の長さ。短い返事なら None（相槌なし）。

        候補の {greeting} は時間帯の挨拶（src/timeofday.py。応答・定型文と同じ判定）で埋める。
        """
        t = text.strip()
        if not t or any(p.search(t) for p in self._none):
            return None
        cat = next((c for c in self._categories if any(p.search(t) for p in c.patterns)), self._default)
        if not cat.candidates:
            return None
        word = cat.candidates[cat.turn % len(cat.candidates)]
        cat.turn += 1
        return word.replace("{greeting}", greeting or greeting_for()), cat.pause_sec

    def all_variants(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for c in [*self._categories, self._default]:
            for w in c.candidates:
                # {greeting} は時間帯の 3 通りとも先に合成しておく
                words = [w.replace("{greeting}", g) for g in all_greetings()] if "{greeting}" in w else [w]
                out += [(x, c.pause_sec) for x in words]
        return list(dict.fromkeys(out))

    async def prepare(self, tts: Synthesizer, time_budget_sec: float) -> int:
        """全候補を事前合成する。time_budget_sec を超えたら残りは使うときに合成する。"""
        t0 = time.monotonic()
        done = 0
        for word, pause in self.all_variants():
            if time.monotonic() - t0 > time_budget_sec:
                logger.warning("相槌の事前合成が %.0f秒を超えたので、残りは使うときに合成します",
                               time_budget_sec)
                break
            if await self._synthesize(tts, word, pause) is not None:
                done += 1
        logger.info("相槌を事前合成しました: %d 個（%.1f 秒）", done, time.monotonic() - t0)
        return done

    async def get(self, tts: Synthesizer, word: str, pause: float) -> BackchannelClip | None:
        clip = self._cache.get((word, pause))
        return clip if clip is not None else await self._synthesize(tts, word, pause)

    async def _synthesize(self, tts: Synthesizer, word: str, pause: float) -> BackchannelClip | None:
        try:
            wav, query = await tts.synthesize_with_query(word)
        except Exception:
            logger.warning("相槌を合成できません: %s", word, exc_info=True)
            return None
        clip = BackchannelClip(word, append_silence(wav, pause), query)
        self._cache[(word, pause)] = clip
        return clip


def append_silence(wav: bytes, seconds: float) -> bytes:
    """WAV の末尾に無音を足す（相槌と本応答の間）。"""
    data, rate = sf.read(io.BytesIO(wav), dtype="float32")
    pad = np.zeros((int(rate * seconds),) + data.shape[1:], dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([data, pad]), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def strip_leading_backchannel(sentence: str, word: str) -> str:
    """本応答の 1 文目が相槌と同じ語で始まるなら、その文頭を落とす（「はい、…」→「…」）。"""
    if not word:
        return sentence
    m = re.match(rf"^\s*{re.escape(word)}[、。,，！!…ー〜]*\s*", sentence)
    return sentence[m.end():] if m else sentence
