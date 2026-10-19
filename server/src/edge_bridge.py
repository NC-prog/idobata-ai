"""Unity WebSocket Bridge.

Live2D アバターとの通信を管理する。websockets v17+ 対応済み。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.asyncio.server import Server, ServerConnection

logger = logging.getLogger(__name__)


class UnityBridge:
    """Unity との WebSocket 通信を管理するクラス。"""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.clients: set[ServerConnection] = set()
        self.server: Server | None = None
        self._running = False

    async def start(self) -> None:
        """WebSocket サーバーを開始する。"""
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )
        self._running = True
        logger.info("Unity Bridge started on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """WebSocket サーバーを停止する。"""
        self._running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("Unity Bridge stopped")

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """クライアント接続を処理する。"""
        self.clients.add(websocket)
        client_info = str(websocket.remote_address)
        logger.info("Unity connected: %s", client_info)

        try:
            async for message in websocket:
                logger.debug("Received from Unity: %s", message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Unity disconnected: %s", client_info)
        finally:
            self.clients.discard(websocket)

    async def send_command(self, command: dict[str, Any]) -> None:
        """全クライアントにコマンドを送信する。"""
        if not self.clients:
            logger.warning("No Unity clients connected")
            return

        message = json.dumps(command, ensure_ascii=False)

        disconnected: set[ServerConnection] = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        self.clients -= disconnected

    # ========== 既存コマンド（type ベース、互換維持） ==========

    async def talk_start(self) -> None:
        """発話開始（リップシンク開始）。"""
        await self.send_command({"type": "talk_start"})
        logger.info("Sent: talk_start")

    async def talk_end(self) -> None:
        """発話終了（リップシンク停止）。"""
        await self.send_command({"type": "talk_end"})
        logger.info("Sent: talk_end")

    async def bow(self) -> None:
        """お辞儀モーション。"""
        await self.send_command({"type": "motion", "name": "bow"})
        logger.info("Sent: bow motion")

    async def celebrate(self) -> None:
        """祝福モーション。"""
        await self.send_command({"type": "motion", "name": "celebrate"})
        logger.info("Sent: celebrate motion")

    async def set_expression(self, value: float) -> None:
        """表情設定 (0.0 = 通常, 1.0 = 笑顔)。"""
        await self.send_command({"type": "expression", "value": value})
        logger.info("Sent: expression %s", value)

    async def set_lipsync(self, value: float) -> None:
        """リップシンク値を直接設定 (0.0 - 1.0)。"""
        await self.send_command({"type": "lipsync", "value": value})

    # ========== 新コマンド（command ベース） ==========

    async def send_caption(self, text: str, *, final: bool = False) -> None:
        """字幕テキストを送信する。

        Args:
            text: 字幕テキスト
            final: 確定テキストかどうか
        """
        await self.send_command({
            "command": "caption",
            "text": text,
            "final": final,
        })

    async def show_panel(self, panel: str, markdown: str) -> None:
        """パネルを表示する。

        Args:
            panel: パネル種別（"summary" 等）
            markdown: 表示する Markdown テキスト
        """
        await self.send_command({
            "command": "show_panel",
            "panel": panel,
            "markdown": markdown,
        })
        logger.info("Sent: show_panel (%s)", panel)

    async def hide_panel(self) -> None:
        """パネルを非表示にする。"""
        await self.send_command({"command": "hide_panel"})
        logger.info("Sent: hide_panel")

    async def set_presenter(self, name: str, dept: str) -> None:
        """現在の発表者を設定する。

        Args:
            name: 発表者名
            dept: 部署名
        """
        await self.send_command({
            "command": "set_presenter",
            "name": name,
            "dept": dept,
        })
        logger.info("Sent: set_presenter %s (%s)", name, dept)

    @property
    def is_connected(self) -> bool:
        """Unity が接続されているか。"""
        return len(self.clients) > 0


# グローバルインスタンス（シングルトン的に使用）
_bridge_instance: UnityBridge | None = None


def get_bridge() -> UnityBridge:
    """UnityBridge のグローバルインスタンスを取得する。"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = UnityBridge()
    return _bridge_instance


async def test_bridge() -> None:
    """テスト用: コマンドラインからの動作確認。"""
    bridge = get_bridge()
    await bridge.start()

    print("\n=== Unity Bridge Test Server ===")
    print("Unity からの接続を待っています...")
    print("コマンド: talk, stop, bow, celebrate, smile, normal, quit")
    print("=" * 35)

    async def input_loop() -> None:
        loop = asyncio.get_event_loop()
        while True:
            try:
                command = await loop.run_in_executor(None, input, "\n> ")
                command = command.strip().lower()

                if command in ("quit", "q"):
                    print("終了します...")
                    break
                elif command == "talk":
                    await bridge.talk_start()
                    print("-> リップシンク開始")
                elif command == "stop":
                    await bridge.talk_end()
                    print("-> リップシンク停止")
                elif command == "bow":
                    await bridge.bow()
                    print("-> お辞儀")
                elif command == "celebrate":
                    await bridge.celebrate()
                    print("-> 祝福")
                elif command == "smile":
                    await bridge.set_expression(1.0)
                    print("-> 笑顔")
                elif command == "normal":
                    await bridge.set_expression(0.0)
                    print("-> 通常表情")
                elif command == "status":
                    status = "接続中" if bridge.is_connected else "未接続"
                    print(f"-> Unity: {status} ({len(bridge.clients)} clients)")
                else:
                    print("不明なコマンド。talk/stop/bow/celebrate/smile/normal/status/quit")

            except EOFError:
                break

    await input_loop()
    await bridge.stop()


if __name__ == "__main__":
    asyncio.run(test_bridge())
