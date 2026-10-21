"""エッジページ（ブラウザのアバター画面）の静的配信（WP01）。

``edge/`` を Vite でビルドした成果物（``server/static/edge/``）を
``http://<host>:<port>/edge/`` で配信する。WebSocket はここではなく edge_bridge（8765）。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

_NOT_BUILT_HTML = """<!doctype html><meta charset="utf-8"><title>井戸端AI エッジ</title>
<p>エッジページがビルドされていません。<code>cd edge &amp;&amp; npm ci &amp;&amp; npm run build</code>
を実行してから再読み込みしてください。</p>"""


def create_app(static_dir: Path) -> FastAPI:
    """エッジページを配信する FastAPI アプリ。"""
    app = FastAPI(title="idobata-edge", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/edge/")

    if (static_dir / "index.html").is_file():
        app.mount("/edge", StaticFiles(directory=static_dir, html=True), name="edge")
    else:
        logger.warning("エッジページが未ビルドです: %s（edge/ で npm run build）", static_dir)

        @app.get("/edge/", include_in_schema=False)
        async def not_built() -> HTMLResponse:
            return HTMLResponse(_NOT_BUILT_HTML, status_code=503)

    return app


class EdgeServer:
    """uvicorn を同じイベントループで動かす。"""

    def __init__(self, host: str, port: int, static_dir: Path) -> None:
        cfg = uvicorn.Config(
            create_app(static_dir), host=host, port=port, log_level="warning", lifespan="off",
        )
        self._server = uvicorn.Server(cfg)
        self._task: asyncio.Task[None] | None = None
        self.url = f"http://{host}:{port}/edge/"

    async def start(self) -> None:
        """配信を開始する（待ち受けが始まるまで待つ）。"""
        self._task = asyncio.create_task(self._server.serve(), name="edge-http")
        while not self._server.started:
            if self._task.done():
                self._task.result()  # 起動失敗（ポート使用中など）をそのまま送出する
            await asyncio.sleep(0.05)
        logger.info("エッジページ: %s", self.url)

    async def stop(self) -> None:
        """配信を止める。"""
        self._server.should_exit = True
        if self._task is not None:
            await self._task
