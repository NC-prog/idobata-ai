"""エッジページ（ブラウザのアバター画面）の静的配信（WP01）と、職員パネル（WP04 B-2）。

``edge/`` を Vite でビルドした成果物（``server/static/edge/``）を
``http://<host>:<port>/edge/`` で配信する。WebSocket はここではなく edge_bridge（8765）。

職員パネル ``/staff/``: 紹介状の一覧と状態の変更、今日の状況。LAN 内（プライベートアドレス）からだけ、
Basic 認証（ユーザー staff、パスワードは .env の IDOBATA_STAFF_PASSWORD。未設定なら使えない）。

管理者ページ ``/admin/transcripts/``（WP06）: 会話の文字記録を日付・セッションごとに読み、保全を付け外しする。
職員パネルとは別のロール（ユーザー admin、IDOBATA_ADMIN_PASSWORD。未設定なら使えない）。LAN 内からだけ。
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import AdminConfig, StaffConfig
from .edge_bridge import allowed_peer
from .knowledge import JST
from .procedures import AsrFixes
from .referrals import ReferralStore, TidbitStore, VisitLog, today_stats
from .transcripts import TranscriptStore

logger = logging.getLogger(__name__)

_NOT_BUILT_HTML = """<!doctype html><meta charset="utf-8"><title>井戸端AI エッジ</title>
<p>エッジページがビルドされていません。<code>cd edge &amp;&amp; npm ci &amp;&amp; npm run build</code>
を実行してから再読み込みしてください。</p>"""


_STAFF_HTML = Path(__file__).resolve().parent.parent / "static" / "staff.html"
_ADMIN_HTML = Path(__file__).resolve().parent.parent / "static" / "admin_transcripts.html"


class StatusUpdate(BaseModel):
    status: str | None = None
    needs_action: bool | None = None
    handled_by_role: str | None = None
    resolution: str | None = None
    department: str | None = None


class HoldIn(BaseModel):
    held: bool
    note: str = ""


class AsrFixIn(BaseModel):
    surface: str
    correct: str


class StaffPanel:
    """職員パネルの中身（紹介状・来訪記録）。"""

    def __init__(self, referrals: ReferralStore, visits: VisitLog, cfg: StaffConfig,
                 tidbits: TidbitStore | None = None, departments: tuple[str, ...] = (),
                 asr_fixes: AsrFixes | None = None) -> None:
        self.referrals = referrals
        self.visits = visits
        self.cfg = cfg
        self.tidbits = tidbits
        # 職員が「未特定」の紹介状に付けられる課（拠点 YAML の facility.departments）
        self.departments = departments
        self.asr_fixes = asr_fixes


class AdminPanel:
    """管理者ページの中身（会話の文字記録）。"""

    def __init__(self, transcripts: TranscriptStore, cfg: AdminConfig) -> None:
        self.transcripts = transcripts
        self.cfg = cfg


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


def _basic_guard(realm: str, user: str, password: str, what: str, env_name: str) -> Any:
    """LAN 内からだけ・Basic 認証の依存関係（職員パネルと管理者ページで別のユーザー・パスワード）。"""
    basic = HTTPBasic(realm=realm)

    # Depends は既定値の形で書く（関数の中の basic を Annotated で参照すると、注釈の遅延評価で解決できない）
    def guard(request: Request, cred: HTTPBasicCredentials = Depends(basic)) -> None:  # noqa: B008
        if not _is_lan(request.client.host if request.client else None):
            raise HTTPException(403, f"{what}は LAN 内からだけ使えます")
        if not password:
            raise HTTPException(503, f"{env_name} が設定されていません")
        ok_user = secrets.compare_digest(cred.username.encode(), user.encode())
        ok_pass = secrets.compare_digest(cred.password.encode(), password.encode())
        if not (ok_user and ok_pass):
            raise HTTPException(401, "認証に失敗しました", headers={"WWW-Authenticate": "Basic"})

    return Depends(guard)


def _add_admin_routes(app: FastAPI, panel: AdminPanel) -> None:
    auth = [_basic_guard("idobata-admin", panel.cfg.user, panel.cfg.password, "管理者ページ",
                         "IDOBATA_ADMIN_PASSWORD")]

    @app.get("/admin/transcripts/", include_in_schema=False, dependencies=auth)
    async def transcripts_page() -> HTMLResponse:
        html = _ADMIN_HTML.read_text(encoding="utf-8").replace(
            "__RETENTION_DAYS__", str(panel.transcripts.retention_days))
        return HTMLResponse(html)

    @app.get("/admin/api/transcripts/days", dependencies=auth)
    async def transcript_days() -> list[str]:
        return panel.transcripts.days()

    @app.get("/admin/api/transcripts", dependencies=auth)
    async def transcript_sessions(day: str | None = None) -> list[dict[str, Any]]:
        return panel.transcripts.sessions(_parse_day(day))

    @app.get("/admin/api/transcripts/{day}/{session_id}", dependencies=auth)
    async def transcript_session(day: str, session_id: str) -> list[dict[str, Any]]:
        rows = panel.transcripts.session(_parse_day(day), session_id)
        if not rows:
            raise HTTPException(404, "記録がありません")
        return rows

    @app.post("/admin/api/transcripts/{session_id}/hold", dependencies=auth)
    async def transcript_hold(session_id: str, body: HoldIn) -> dict[str, bool]:
        if not panel.transcripts.set_hold(session_id, body.held, body.note):
            raise HTTPException(404, "記録がありません")
        return {"held": body.held}


def _add_staff_routes(app: FastAPI, panel: StaffPanel) -> None:
    auth = [_basic_guard("idobata-staff", panel.cfg.user, panel.cfg.password, "職員パネル",
                         "IDOBATA_STAFF_PASSWORD")]

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
                                     handled_by_role=body.handled_by_role, resolution=body.resolution,
                                     department=body.department, departments=panel.departments)
        if ref is None:
            raise HTTPException(404, "紹介状がありません")
        return asdict(ref)

    @app.get("/staff/api/departments", dependencies=auth)
    async def list_departments() -> list[str]:
        return list(panel.departments)

    @app.get("/staff/api/asr_fixes", dependencies=auth)
    async def list_asr_fixes() -> list[dict[str, str]]:
        return panel.asr_fixes.entries() if panel.asr_fixes is not None else []

    @app.post("/staff/api/asr_fixes", dependencies=auth)
    async def add_asr_fix(body: AsrFixIn) -> dict[str, str]:
        if panel.asr_fixes is None:
            raise HTTPException(503, "聞き取り間違いの対応表が使えません")
        try:
            return panel.asr_fixes.add(body.surface, body.correct)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

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


_DEBUG_HTML = """<!doctype html><meta charset="utf-8"><title>井戸端AI カメラ（デバッグ）</title>
<body style="margin:0;background:#111;color:#ccc;font:14px sans-serif">
<p style="margin:8px">検出の矩形とキーポイント（メモリ上の最新の 1 枚だけ。保存しない）。<span id="s"></span></p>
<img id="f" style="width:640px;image-rendering:pixelated">
<script>
const f = document.getElementById("f"), s = document.getElementById("s");
setInterval(async () => {
  const r = await fetch("/edge/debug/party"); if (r.ok) s.textContent = JSON.stringify(await r.json());
  f.src = "/edge/debug.jpg?t=" + Date.now();
}, 300);
</script>"""

DebugFrame = Callable[[], bytes | None]
DebugState = Callable[[], dict[str, Any]]


def _add_debug_routes(app: FastAPI, frame: DebugFrame, state: DebugState) -> None:
    """--camera-debug のときだけ: 検出結果を重ねた最新の 1 枚（メモリのみ）。LAN 内からだけ。"""
    def lan_only(request: Request) -> None:
        if not _is_lan(request.client.host if request.client else None):
            raise HTTPException(403, "LAN 内からだけ使えます")

    guard = [Depends(lan_only)]

    @app.get("/edge/debug", include_in_schema=False, dependencies=guard)
    async def debug_page() -> HTMLResponse:
        return HTMLResponse(_DEBUG_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/edge/debug.jpg", include_in_schema=False, dependencies=guard)
    async def debug_jpeg() -> Response:
        data = frame()
        if data is None:
            raise HTTPException(404, "まだフレームがありません")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/edge/debug/party", include_in_schema=False, dependencies=guard)
    async def debug_party() -> dict[str, Any]:
        return state()


def create_app(static_dir: Path, staff: StaffPanel | None = None, admin: AdminPanel | None = None,
               debug_frame: DebugFrame | None = None, debug_state: DebugState | None = None,
               edge_allow: tuple[str, ...] | None = None) -> FastAPI:
    """エッジページ（と職員パネル・管理者ページ・カメラのデバッグ表示）を配信する FastAPI アプリ。

    edge_allow を渡すと、/edge/ と / はこの機体自身と edge_allow の相手からだけ開ける（IDOBATA_EDGE_ALLOW。WP02 D）。
    """
    app = FastAPI(title="idobata-edge", docs_url=None, redoc_url=None, openapi_url=None)
    if edge_allow is not None:
        allow = edge_allow

        @app.middleware("http")
        async def only_allowed_edges(request: Request, call_next: Any) -> Any:
            path = request.url.path
            if (path == "/" or path.startswith("/edge")) and not allowed_peer(
                    request.client.host if request.client else None, allow):
                return Response("not allowed", status_code=403)
            return await call_next(request)
    if debug_frame is not None:
        _add_debug_routes(app, debug_frame, debug_state or (dict))  # /edge の静的配信より先に置く
    if staff is not None:
        _add_staff_routes(app, staff)
    if admin is not None:
        _add_admin_routes(app, admin)

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

    def __init__(self, host: str, port: int, static_dir: Path, staff: StaffPanel | None = None,
                 admin: AdminPanel | None = None, debug_frame: DebugFrame | None = None,
                 debug_state: DebugState | None = None, edge_allow: tuple[str, ...] | None = None) -> None:
        cfg = uvicorn.Config(
            create_app(static_dir, staff, admin, debug_frame, debug_state, edge_allow), host=host, port=port, log_level="warning", lifespan="off",
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
