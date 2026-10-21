"""エッジ（ブラウザ）⇄ フロントAI の WebSocket ブリッジ（WP01）。

旧 unity_bridge。1 本の WebSocket で JSON（テキストフレーム）と音声（バイナリフレーム）を
混在させる。契約は docs/WP01_edge_and_avatar.md B-1 と docs/WP01_result.md。

上り（エッジ → フロント）
- JSON ``hello``: ``{type, edge_id, sample_rate, caps:{aec}}``
- バイナリ: PCM16 mono 16kHz（20ms ごと）→ ``on_pcm`` に渡す（EdgeAudioInput）
- JSON ``playback_ended``: ``{type, utterance_id, last_seq}`` 再生キューが空になった
  （``last_seq`` は最後に鳴らし終えた文の番号。契約への追加。文と文の間で一瞬キューが
  空になった通知と、応答の最後の通知を見分けるのに使う）
- JSON ``interrupt``: 予約（WP03）

下り（フロント → エッジ）
- JSON ``state`` / ``caption`` / ``theme``
- JSON ``speak`` + 直後にバイナリ（その文の WAV）

音声を送ってくるエッジは 1 台（最後に hello したもの）とする。表示用の下りは全接続に送る。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection

logger = logging.getLogger(__name__)

# エッジの状態（アバターの見た目）
STATES = ("idle", "listening", "thinking", "speaking")

PcmHandler = Callable[[bytes], None]
HelloHandler = Callable[[dict[str, Any]], None]


@dataclass
class _PlaybackWaiter:
    utterance_id: str
    last_seq: int
    future: asyncio.Future[None] = field(repr=False)


class EdgeBridge:
    """エッジとの WebSocket サーバ。"""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.clients: set[ServerConnection] = set()
        self.server: Server | None = None
        self._audio_client: ServerConnection | None = None
        self._send_lock = asyncio.Lock()
        self._waiters: list[_PlaybackWaiter] = []
        # 受信したうち、各発話で鳴らし終えた最大の文番号（待ちより先に届いた通知用）
        self._played: dict[str, int] = {}
        self.on_pcm: PcmHandler | None = None
        self.on_hello: HelloHandler | None = None

    async def start(self) -> None:
        """WebSocket サーバーを開始する。"""
        self.server = await websockets.serve(
            self._handle_client, self.host, self.port, max_size=2**22,
        )
        logger.info("Edge Bridge: ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """WebSocket サーバーを停止する。"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        for w in self._waiters:
            if not w.future.done():
                w.future.cancel()
        logger.info("Edge Bridge stopped")

    @property
    def is_connected(self) -> bool:
        """音声を送ってくるエッジが接続しているか。"""
        return self._audio_client is not None

    # ------------------------------------------------------------------
    # 受信
    # ------------------------------------------------------------------

    async def _handle_client(self, websocket: ServerConnection) -> None:
        self.clients.add(websocket)
        peer = str(websocket.remote_address)
        logger.info("エッジ接続: %s", peer)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if websocket is self._audio_client and self.on_pcm is not None:
                        self.on_pcm(message)
                    continue
                self._handle_json(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            if websocket is self._audio_client:
                self._audio_client = None
            logger.info("エッジ切断: %s", peer)

    def _handle_json(self, websocket: ServerConnection, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("エッジから JSON でないテキストを受信しました")
            return
        if not isinstance(data, dict):
            return
        kind = data.get("type")
        if kind == "hello":
            self._audio_client = websocket
            logger.info(
                "エッジ hello: edge_id=%s sample_rate=%s caps=%s",
                data.get("edge_id"), data.get("sample_rate"), data.get("caps"),
            )
            if self.on_hello is not None:
                self.on_hello(data)
        elif kind == "playback_ended":
            self._on_playback_ended(str(data.get("utterance_id", "")), data.get("last_seq"))
        elif kind == "interrupt":
            logger.info("エッジ interrupt（WP03 まで未対応）")
        else:
            logger.debug("未知のメッセージ: %s", kind)

    def _on_playback_ended(self, utterance_id: str, last_seq: Any) -> None:
        seq = last_seq if isinstance(last_seq, int) else -1
        logger.debug("playback_ended: %s last_seq=%d", utterance_id, seq)
        self._played[utterance_id] = max(seq, self._played.get(utterance_id, -1))
        for w in list(self._waiters):
            if w.utterance_id == utterance_id and seq >= w.last_seq and not w.future.done():
                w.future.set_result(None)

    async def wait_playback_ended(self, utterance_id: str, last_seq: int, timeout: float) -> bool:
        """エッジが utterance_id の last_seq 番目の文まで鳴らし終えるのを待つ。

        Returns:
            True: 通知が来た / False: timeout 秒以内に来なかった（エッジの切断など）
        """
        if self._played.get(utterance_id, -1) >= last_seq:
            self._played.pop(utterance_id, None)
            return True
        loop = asyncio.get_running_loop()
        waiter = _PlaybackWaiter(utterance_id, last_seq, loop.create_future())
        self._waiters.append(waiter)
        try:
            await asyncio.wait_for(waiter.future, timeout)
            return True
        except TimeoutError:
            return False
        finally:
            self._waiters.remove(waiter)
            self._played.pop(utterance_id, None)

    # ------------------------------------------------------------------
    # 送信
    # ------------------------------------------------------------------

    async def send_json(self, payload: dict[str, Any]) -> None:
        """全接続に JSON を送る。"""
        await self._broadcast([json.dumps(payload, ensure_ascii=False)])

    async def send_state(self, value: str) -> None:
        """アバターの状態（idle / listening / thinking / speaking）。"""
        if value not in STATES:
            raise ValueError(f"未知の状態: {value}")
        await self.send_json({"type": "state", "value": value})

    async def send_caption(self, text: str, *, final: bool, role: str = "user") -> None:
        """字幕（caption_feed からも使う）。"""
        await self.send_json({"type": "caption", "role": role, "text": text, "final": final})

    async def send_theme(self, site: str, season: str) -> None:
        """衣装・背景の切替（WP01 は背景だけ）。"""
        await self.send_json({"type": "theme", "site": site, "season": season})

    async def send_speak(self, header: dict[str, Any], wav: bytes) -> None:
        """1 文ぶんの ``speak`` と WAV を、間に他の送信が挟まらないよう続けて送る。"""
        await self._broadcast([json.dumps(header, ensure_ascii=False), wav])

    async def _broadcast(self, frames: list[str | bytes]) -> None:
        if not self.clients:
            return
        async with self._send_lock:
            for client in list(self.clients):
                try:
                    for frame in frames:
                        await client.send(frame)
                except websockets.exceptions.ConnectionClosed:
                    self.clients.discard(client)
