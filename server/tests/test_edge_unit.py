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
        frames: list[bytes] = []
        bridge.on_pcm = pcm.append
        bridge.on_video = frames.append
        async with websockets.connect(url) as ws:
            await ws.send(b"\x01\x00\x01")  # hello 前の音声は聞かない
            await ws.send(json.dumps({"type": "hello", "edge_id": "e1", "sample_rate": 16000,
                                      "caps": {"aec": True}}))
            await _until(lambda: bridge.is_connected)
            await ws.send(b"\x01\x02\x03")  # 先頭 1 バイトが種別（0x01 = PCM、0x02 = JPEG）
            await ws.send(b"\x02\xff\xd8")
            await _until(lambda: pcm == [b"\x02\x03"] and frames == [b"\xff\xd8"])
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

    async def test_second_edge_replaces_first(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from src.edge_bridge import REPLACED_CLOSE_CODE

        bridge, url = await _started_bridge()
        hello = json.dumps({"type": "hello", "edge_id": "e", "sample_rate": 16000, "caps": {}})
        pcm: list[bytes] = []
        bridge.on_pcm = pcm.append
        async with websockets.connect(url) as first, websockets.connect(url) as second:
            await first.send(hello)
            await _until(lambda: bridge.is_connected)
            with caplog.at_level(logging.WARNING, logger="src.edge_bridge"):
                await second.send(hello)
                with pytest.raises(websockets.exceptions.ConnectionClosed) as exc:
                    await asyncio.wait_for(first.recv(), 2.0)
            assert exc.value.rcvd is not None and exc.value.rcvd.code == REPLACED_CLOSE_CODE
            assert any("別のエッジ" in r.message for r in caplog.records)
            await second.send(b"\x01\x01\x02")
            await _until(lambda: pcm == [b"\x01\x02"])
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


def test_allowed_peer_for_remote_edge() -> None:
    from src.edge_bridge import allowed_peer

    assert allowed_peer("127.0.0.1", ())  # この機体自身はいつでも
    assert not allowed_peer("192.168.10.21", ())  # 空なら、この機体からだけ
    assert allowed_peer("192.168.10.21", ("192.168.10.21",))
    assert allowed_peer("10.0.0.7", ("192.168.10.21", "10.0.0.0/24"))
    assert not allowed_peer("192.168.10.22", ("192.168.10.21",))
    assert not allowed_peer("192.168.10.22", ("bad",))


async def test_edge_page_only_for_allowed_edges(tmp_path: Any) -> None:
    from src.edge_server import create_app

    (tmp_path / "index.html").write_text("<p>edge</p>", encoding="utf-8")
    app = create_app(tmp_path, edge_allow=("192.168.10.21",))
    for client, code in ((("192.168.10.21", 5000), 200), (("192.168.10.99", 5000), 403), (("127.0.0.1", 5000), 200)):
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            assert (await c.get("/edge/")).status_code == code, client
