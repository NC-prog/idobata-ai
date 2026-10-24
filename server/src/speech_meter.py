"""要約などの読み上げの長さを VOICEVOX で測る（夜のお知らせの要約と、職員パネルの「確認済みとして保存」。WP11・WP13）。"""

from __future__ import annotations

import io
import re
import time
import wave

import httpx

from .config import config
from .dialogue import speech_text


class SpeechMeter:
    """VOICEVOX で合成した長さ（秒）。文ごとに合成して足す（読み上げも文単位で合成する）。"""

    def __init__(self) -> None:
        self.cfg = config.voicevox
        self.client = httpx.Client(base_url=self.cfg.base_url, timeout=self.cfg.timeout)

    def _post(self, path: str, **kwargs: object) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.client.post(path, **kwargs)  # type: ignore[arg-type]
                r.raise_for_status()
                return r
            except httpx.HTTPError as exc:
                last = exc
                time.sleep(self.cfg.retry_wait_sec * (attempt + 1))
        raise RuntimeError(f"VOICEVOX に繋がりません: {last}")

    def seconds(self, text: str) -> float:
        total = 0.0
        for sentence in re.split(r"(?<=[。！？!?])", text):
            spoken = speech_text(sentence)
            if not spoken:
                continue
            q = self._post("/audio_query", params={"text": spoken, "speaker": self.cfg.speaker_id}).json()
            q.update(speedScale=self.cfg.speed_scale, pitchScale=self.cfg.pitch_scale,
                     intonationScale=self.cfg.intonation_scale, volumeScale=self.cfg.volume_scale,
                     postPhonemeLength=self.cfg.post_phoneme_length)
            wav = self._post("/synthesis", params={"speaker": self.cfg.speaker_id}, json=q).content
            with wave.open(io.BytesIO(wav)) as w:
                total += w.getnframes() / w.getframerate()
        return round(total, 1)

    def close(self) -> None:
        self.client.close()
