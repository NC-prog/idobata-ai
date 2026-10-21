"""エッジ連携（edge_bridge / EdgeAudioInput / EdgeSpeaker / edge_server）のテスト。"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]
import websockets

from src.audio_input import EdgeAudioInput
from src.config import AudioInputConfig
from src.edge_bridge import EdgeBridge
from src.edge_server import create_app
from src.main import current_season
from src.speech_output import EdgeSpeaker, SpokenSentence


async def _started_bridge() -> tuple[EdgeBridge, str]:
    bridge = EdgeBridge("127.0.0.1", 0)
    await bridge.start()
    assert bridge.server is not None
    port = next(iter(bridge.server.sockets)).getsockname()[1]
    return bridge, f"ws://127.0.0.1:{port}"


async def _until(cond: Any, timeout: float = 2.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("条件が満たされませんでした")


class TestEdgeBridge:
    async def test_hello_pcm_and_downlink(self) -> None:
        bridge, url = await _started_bridge()
        hellos: list[dict[str, Any]] = []
        pcm: list[bytes] = []
        bridge.on_hello = hellos.append
        bridge.on_pcm = pcm.append
        async with websockets.connect(url) as ws:
            await ws.send(b"\x00\x01")  # hello 前の音声は聞かない
            await ws.send(json.dumps({"type": "hello", "edge_id": "e1", "sample_rate": 16000,
                                      "caps": {"aec": True}}))
            await _until(lambda: bridge.is_connected)
            await ws.send(b"\x02\x03")
            await _until(lambda: pcm == [b"\x02\x03"])
            assert hellos[0]["edge_id"] == "e1"

            await bridge.send_state("thinking")
            assert json.loads(await ws.recv()) == {"type": "state", "value": "thinking"}
            await bridge.send_speak({"type": "speak", "seq": 0}, b"RIFF")
            assert json.loads(await ws.recv())["type"] == "speak"
            assert await ws.recv() == b"RIFF"
        await _until(lambda: not bridge.is_connected)
        await bridge.stop()

    async def test_wait_playback_ended_ignores_earlier_sentences(self) -> None:
        bridge, url = await _started_bridge()
        async with websockets.connect(url) as ws:
            waiter = asyncio.create_task(bridge.wait_playback_ended("u1", 2, timeout=2.0))
            await asyncio.sleep(0.05)
            # 文と文の間でキューが空になった通知（seq 1）では終わらない
            await ws.send(json.dumps({"type": "playback_ended", "utterance_id": "u1",
                                      "last_seq": 1}))
            await asyncio.sleep(0.1)
            assert not waiter.done()
            await ws.send(json.dumps({"type": "playback_ended", "utterance_id": "u1",
                                      "last_seq": 2}))
            assert await waiter is True
        await bridge.stop()

    async def test_wait_playback_ended_arrived_early_and_timeout(self) -> None:
        bridge, url = await _started_bridge()
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "playback_ended", "utterance_id": "u2",
                                      "last_seq": 0}))
            await asyncio.sleep(0.1)
            assert await bridge.wait_playback_ended("u2", 0, timeout=0.5) is True
            assert await bridge.wait_playback_ended("u3", 0, timeout=0.1) is False
        await bridge.stop()

    async def test_unknown_state_rejected(self) -> None:
        with pytest.raises(ValueError):
            await EdgeBridge("127.0.0.1", 0).send_state("dancing")


async def test_edge_audio_input_rechunks_to_vad_frames() -> None:
    src = EdgeAudioInput(AudioInputConfig(frame_size=512))
    chunk = (np.full(320, 16384, dtype="<i2")).tobytes()  # 0.5
    src.push_pcm(chunk)
    src.push_pcm(chunk)  # 640 サンプル → 512 のフレーム 1 つと端数 128
    frame = await asyncio.wait_for(src.__anext__(), 1.0)
    assert frame.shape == (512,)
    assert np.allclose(frame, 0.5)
    await src.stop()
    with pytest.raises(StopAsyncIteration):
        await src.__anext__()


class _FakeBridge:
    def __init__(self, ended: bool = True) -> None:
        self.sent: list[tuple[str, Any]] = []
        self.ended = ended

    async def send_state(self, value: str) -> None:
        self.sent.append(("state", value))

    async def send_speak(self, header: dict[str, Any], wav: bytes) -> None:
        self.sent.append(("speak", header))

    async def send_caption(self, text: str, *, final: bool, role: str = "user") -> None:
        self.sent.append(("caption", (role, text, final)))

    async def wait_playback_ended(self, utterance_id: str, last_seq: int, timeout: float) -> bool:
        self.sent.append(("wait", (utterance_id, last_seq)))
        return self.ended


def _wav(sec: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(24000 * sec), dtype=np.float32), 24000, format="WAV")
    return buf.getvalue()


async def test_edge_speaker_sends_speak_and_waits_for_last_sentence() -> None:
    bridge = _FakeBridge()
    speaker = EdgeSpeaker(bridge, timeout_margin_sec=1.0)  # type: ignore[arg-type]
    query = {"prePhonemeLength": 0.1, "accent_phrases": [
        {"moras": [{"consonant_length": None, "vowel": "a", "vowel_length": 0.1}]}]}
    r0 = await speaker.push(SpokenSentence("u7", 0, "あ。", _wav(0.5), query))
    r1 = await speaker.push(SpokenSentence("u7", 1, "い。", _wav(0.5), {}))
    await speaker.finish()
    assert 0.45 < r0 <= 0.5 and 0.95 < r1 <= 1.0
    kinds = [k for k, _ in bridge.sent]
    assert kinds[0] == "state" and bridge.sent[0][1] == "speaking"
    speak0 = bridge.sent[1][1]
    assert speak0["utterance_id"] == "u7" and speak0["seq"] == 0
    assert speak0["mouth"] == [{"t": 0.1, "v": "aa", "d": 0.1}]
    assert ("caption", ("assistant", "あ。い。", True)) in bridge.sent
    assert ("wait", ("u7", 1)) in bridge.sent
    assert bridge.sent[-1] == ("state", "idle")


@pytest.mark.parametrize(("month", "season"), [
    (1, "winter"), (2, "winter"), (3, "spring"), (5, "spring"), (6, "summer"),
    (9, "autumn"), (11, "autumn"), (12, "winter"),
])
def test_current_season(month: int, season: str) -> None:
    assert current_season(month) == season


async def test_edge_server_serves_built_page(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<p>edge</p>", encoding="utf-8")
    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/edge/")
        assert r.status_code == 200 and "edge" in r.text
        assert (await client.get("/")).headers["location"] == "/edge/"


async def test_edge_server_reports_missing_build(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(tmp_path / "none"))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/edge/")
        assert r.status_code == 503 and "npm run build" in r.text
