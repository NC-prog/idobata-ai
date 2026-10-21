"""実サービス（ASR / LLM / VOICEVOX）に繋ぐ往復テスト。

``docker compose -f deploy/standalone/docker-compose.yaml up -d`` 後に
``uv run pytest -m integration`` で実行する。
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.asr_client import ASRClient, load_site_vocabulary
from src.config import config
from src.dialogue import Dialogue
from src.llm_client import LLMClient
from src.tts_client import TTSClient

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"


def _load_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr)).astype(np.int64)
        data = data[idx]
    return np.asarray(data, dtype=np.float32)


async def test_wav_asr_llm_tts_roundtrip() -> None:
    asr = ASRClient()
    asr.set_static_context(load_site_vocabulary(config.site, config.asr.site_context_dir))
    tts = TTSClient()
    try:
        audio = _load_16k(FIXTURES / "visitor_procedure_ja.wav")
        final = None
        async for chunk in asr.transcribe_segment([audio], time.monotonic()):
            if chunk.is_final:
                final = chunk
        assert final is not None
        assert "住民票" in final.text
        assert final.language == "ja"

        sentences = [s async for s in Dialogue(LLMClient()).respond(final.text, final.language)]
        assert sentences
        reply = "".join(sentences)
        assert "<think>" not in reply

        wav = await tts.synthesize(reply)
        data, sr = sf.read(io.BytesIO(wav))
        assert sr > 0 and len(data) > sr  # 1 秒以上の音声
    finally:
        await asr.close()
        await tts.close()


async def test_user_dict_fixes_place_name_reading() -> None:
    """拠点 YAML の読みを登録すると「能美」が「ノウミ」と読まれる。"""
    from src.site_context import load_site_words

    tts = TTSClient()
    try:
        await tts.sync_user_dict(load_site_words("etajima", config.asr.site_context_dir))
        _, query = await tts.synthesize_with_query("能美の秋祭りはいつですか")
        kana = "".join(m["text"] for p in query["accent_phrases"] for m in p["moras"])
        assert kana.startswith("ノウミ")
    finally:
        await tts.close()
