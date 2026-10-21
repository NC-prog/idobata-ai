"""エッジページ（ブラウザのアバター画面）の静的配信（WP01）と、職員パネル（WP04 B-2）。

``edge/`` を Vite でビルドした成果物（``server/static/edge/``）を
``http://<host>:<port>/edge/`` で配信する。WebSocket はここではなく edge_bridge（8765）。

職員パネル ``/staff/``: 紹介状の一覧と状態の変更、今日の状況。LAN 内（プライベートアドレス）からだけ、
Basic 認証（ユーザー staff、パスワードは .env の IDOBATA_STAFF_PASSWORD。未設定なら使えない）。
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import StaffConfig
from .knowledge import JST
from .referrals import ReferralStore, TidbitStore, VisitLog, today_stats

logger = logging.getLogger(__name__)

_NOT_BUILT_HTML = """<!doctype html><meta charset="utf-8"><title>井戸端AI エッジ</title>
<p>エッジページがビルドされていません。<code>cd edge &amp;&amp; npm ci &amp;&amp; npm run build</code>
を実行してから再読み込みしてください。</p>"""


_STAFF_HTML = Path(__file__).resolve().parent.parent / "static" / "staff.html"


class StatusUpdate(BaseModel):
    status: str | None = None
    needs_action: bool | None = None
    handled_by_role: str | None = None
    resolution: str | None = None


class StaffPanel:
    """職員パネルの中身（紹介状・来訪記録）。"""

    def __init__(self, referrals: ReferralStore, visits: VisitLog, cfg: StaffConfig,
                 tidbits: TidbitStore | None = None) -> None:
        self.referrals = referrals
        self.visits = visits
        self.cfg = cfg
        self.tidbits = tidbits


def _is_lan(host: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return host == "testclient"  # テスト用のクライアント
    return ip.is_private or ip.is_loopback or ip.is_link_local


def _parse_day(value: str | None) -> date:
    if not value:
        return datetime.now(JST).date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(400, "日付は YYYY-MM-DD") from exc


def _add_staff_routes(app: FastAPI, panel: StaffPanel) -> None:
    basic = HTTPBasic(realm="idobata-staff")

    # Depends は既定値の形で書く（関数の中の basic を Annotated で参照すると、注釈の遅延評価で解決できない）
    def guard(request: Request, cred: HTTPBasicCredentials = Depends(basic)) -> None:  # noqa: B008
        if not _is_lan(request.client.host if request.client else None):
            raise HTTPException(403, "職員パネルは LAN 内からだけ使えます")
        if not panel.cfg.password:
            raise HTTPException(503, "IDOBATA_STAFF_PASSWORD が設定されていません")
        ok_user = secrets.compare_digest(cred.username.encode(), panel.cfg.user.encode())
        ok_pass = secrets.compare_digest(cred.password.encode(), panel.cfg.password.encode())
        if not (ok_user and ok_pass):
            raise HTTPException(401, "認証に失敗しました", headers={"WWW-Authenticate": "Basic"})

    auth = [Depends(guard)]

    @app.get("/staff/", include_in_schema=False, dependencies=auth)
    async def staff_page() -> HTMLResponse:
        html = _STAFF_HTML.read_text(encoding="utf-8").replace("__POLL_SEC__", str(panel.cfg.poll_sec))
        return HTMLResponse(html)

    @app.get("/staff/api/referrals", dependencies=auth)
    async def list_referrals(day: str | None = None) -> list[dict[str, Any]]:
        return [asdict(r) for r in panel.referrals.list(_parse_day(day))]

    @app.post("/staff/api/referrals/{day}/{ref_id}", dependencies=auth)
    async def update_referral(day: str, ref_id: str, body: StatusUpdate) -> dict[str, Any]:
        ref = panel.referrals.update(ref_id, _parse_day(day), status=body.status,
                                     needs_action=body.needs_action,
                                     handled_by_role=body.handled_by_role, resolution=body.resolution)
        if ref is None:
            raise HTTPException(404, "紹介状がありません")
        return asdict(ref)

    @app.get("/staff/api/tidbits", dependencies=auth)
    async def list_tidbits() -> list[dict[str, Any]]:
        return panel.tidbits.active() if panel.tidbits is not None else []

    @app.delete("/staff/api/tidbits/{tid}", dependencies=auth)
    async def delete_tidbit(tid: str) -> dict[str, bool]:
        if panel.tidbits is None or not panel.tidbits.delete(tid):
            raise HTTPException(404, "小ネタがありません")
        return {"deleted": True}

    @app.get("/staff/api/stats", dependencies=auth)
    async def stats(day: str | None = None) -> dict[str, Any]:
        return today_stats(panel.referrals, panel.visits, _parse_day(day))


def create_app(static_dir: Path, staff: StaffPanel | None = None) -> FastAPI:
    """エッジページ（と職員パネル）を配信する FastAPI アプリ。"""
    app = FastAPI(title="idobata-edge", docs_url=None, redoc_url=None, openapi_url=None)
    if staff is not None:
        _add_staff_routes(app, staff)

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

    def __init__(self, host: str, port: int, static_dir: Path, staff: StaffPanel | None = None) -> None:
        cfg = uvicorn.Config(
            create_app(static_dir, staff), host=host, port=port, log_level="warning", lifespan="off",
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
