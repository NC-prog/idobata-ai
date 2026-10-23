"""WP08: テーマ（拠点×季節のアバター・背景・季節の一言）、暦と手動の切り替え、会話中は切り替えない、
アバター・背景の配信（~/idobata-data から、実写の背景が優先）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.camera import PartySnapshot, PartyState
from src.config import SERVER_ROOT, StaffConfig, ThemeConfig
from src.edge_server import StaffPanel, create_app
from src.openers import load_openers, opener_lines
from src.referrals import ReferralStore, VisitLog
from src.themes import ThemeSelector, load_themes, safe_name
from tests.test_camera_unit import _Edge, _loop

THEMES = SERVER_ROOT / "context" / "themes.yaml"


@pytest.mark.parametrize(("site", "day", "name", "avatar", "background"), [
    ("etajima", date(2026, 10, 22), "autumn", "idochan_office.vrm", "etajima_autumn.jpg"),
    ("etajima", date(2026, 12, 1), "winter", "idochan_office.vrm", "etajima_winter.jpg"),
    ("etajima", date(2027, 1, 15), "winter", "idochan_office.vrm", "etajima_winter.jpg"),
    ("etajima", date(2028, 2, 29), "winter", "idochan_office.vrm", "etajima_winter.jpg"),
    ("etajima", date(2027, 3, 1), "spring", "idochan_office.vrm", "etajima_spring.jpg"),
    ("etajima", date(2027, 8, 31), "summer", "idochan_office.vrm", "etajima_summer.jpg"),
    ("akiota", date(2026, 10, 22), "autumn", "idochan_office.vrm", "akiota_autumn.jpg"),
])
def test_calendar_picks_theme(site: str, day: date, name: str, avatar: str, background: str) -> None:
    theme = load_themes(THEMES, site).auto(day)
    assert theme is not None and (theme.name, theme.avatar, theme.background) == (name, avatar, background)
    assert theme.opener_line  # 季節の一言がある


def test_every_theme_has_a_background_and_festival_avatar() -> None:
    bundled = SERVER_ROOT.parent / "edge" / "public" / "backgrounds"
    for site in ("etajima", "akiota"):
        st = load_themes(THEMES, site)
        assert set(st.themes) == {"spring", "summer", "autumn", "winter", "festival"}
        assert st.themes["festival"].avatar == "idochan_festival.vrm"
        for t in st.themes.values():
            assert (bundled / t.background).is_file(), t.background  # 仮置きの背景がビルドに入る


def test_festival_calendar_rule_wins_and_wraps_year(tmp_path: Path) -> None:
    path = tmp_path / "themes.yaml"
    path.write_text("""
sites:
  x:
    themes:
      autumn: {avatar: a.vrm, background: a.jpg}
      winter: {avatar: a.vrm, background: w.jpg}
      festival: {avatar: f.vrm, background: a.jpg, opener_line: お祭りですね。}
    calendar:
      - {theme: festival, from: "10-20", to: "10-25"}
      - {theme: autumn, from: "09-01", to: "11-30"}
      - {theme: winter, from: "12-01", to: "02-29"}
""", encoding="utf-8")
    st = load_themes(path, "x")
    assert st.auto(date(2026, 10, 22)).name == "festival"  # type: ignore[union-attr]
    assert st.auto(date(2026, 10, 26)).name == "autumn"  # type: ignore[union-attr]
    assert st.auto(date(2027, 1, 2)).name == "winter"  # type: ignore[union-attr]
    assert load_themes(path, "unknown").themes == {}


def test_bad_theme_file_falls_back_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "themes.yaml"
    path.write_text("sites:\n  x:\n    themes:\n      a: {avatar: ../../etc/passwd.vrm, background: a.jpg}\n",
                    encoding="utf-8")
    assert load_themes(path, "x").themes == {}
    assert load_themes(tmp_path / "none.yaml", "x").themes == {}


def test_safe_name() -> None:
    assert safe_name("idochan_office.vrm") and safe_name("etajima_autumn.jpg")
    for bad in ("../x.vrm", ".hidden.vrm", "a/b.jpg", "x.txt", "a..b.vrm", ""):
        assert not safe_name(bad), bad


def test_selector_override_and_status() -> None:
    sel = ThemeSelector(load_themes(THEMES, "etajima"))
    day = date(2026, 10, 22)
    assert sel.wanted(day).name == "autumn"  # type: ignore[union-attr]
    sel.set_override("festival")
    assert sel.wanted(day).name == "festival"  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="未定義"):
        sel.set_override("nope")
    st = sel.status(day)
    assert st["auto"] == "autumn" and st["override"] == "festival" and st["wanted"] == "festival"
    assert st["applied"] is None and {"name": "festival", "label": "お祭り"} in st["choices"]
    sel.set_override(None)
    assert sel.wanted(day).name == "autumn"  # type: ignore[union-attr]


def test_opener_season_line_alternates_with_weather() -> None:
    op = load_openers(SERVER_ROOT / "context" / "openers.yaml")
    season = "すっかり秋らしくなりましたね。"
    assert opener_lines(op, "こんにちは", "晴れ", 18.0, 0, season)[1] == "今日はいいお天気ですね。"
    assert opener_lines(op, "こんにちは", "晴れ", 18.0, 1, season)[1] == season  # 1 会話おきに季節の一言
    assert opener_lines(op, "こんにちは", None, None, 0, season)[1] == season  # 天気が分からなければ季節
    assert opener_lines(op, "こんにちは", "晴れ", 7.0, 1, season)[1] == "今日は冷えますね。"  # 寒暖が優先
    assert len(opener_lines(op, "こんにちは", "晴れ", 18.0, 1, "")) == 3  # 季節の一言が無いテーマ


async def test_media_routes_serve_from_data_dir_and_fall_back(tmp_path: Path) -> None:
    static = tmp_path / "static"
    (static / "backgrounds").mkdir(parents=True)
    (static / "index.html").write_text("<p>edge</p>", encoding="utf-8")
    (static / "backgrounds" / "etajima_autumn.jpg").write_bytes(b"bundled")
    (static / "backgrounds" / "etajima_winter.jpg").write_bytes(b"bundled-winter")
    avatars, photos = tmp_path / "avatars", tmp_path / "backgrounds"
    avatars.mkdir()
    photos.mkdir()
    (avatars / "idochan_office.vrm").write_bytes(b"glTF-office")
    (photos / "etajima_autumn.jpg").write_bytes(b"photo")
    (tmp_path / "secret.vrm").write_bytes(b"secret")
    app = create_app(static, avatars_dir=avatars, backgrounds_dir=photos)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/edge/avatars/idochan_office.vrm")
        assert r.status_code == 200 and r.content == b"glTF-office"
        assert (await c.get("/edge/avatars/idochan_festival.vrm")).status_code == 404
        assert (await c.get("/edge/avatars/..%2Fsecret.vrm")).status_code == 404
        assert (await c.get("/edge/backgrounds/etajima_autumn.jpg")).content == b"photo"  # 実写が優先
        assert (await c.get("/edge/backgrounds/etajima_winter.jpg")).content == b"bundled-winter"  # 仮置き
        assert (await c.get("/edge/")).status_code == 200  # 静的配信はそのまま
    blocked = create_app(static, edge_allow=(), avatars_dir=avatars)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=blocked, client=("192.168.1.50", 5000)),
                                 base_url="http://t") as c:
        assert (await c.get("/edge/avatars/idochan_office.vrm")).status_code == 403  # 許可の無い機体からは取れない


async def test_staff_panel_switches_theme(tmp_path: Path) -> None:
    sel = ThemeSelector(load_themes(THEMES, "etajima"))
    staff = StaffPanel(ReferralStore(tmp_path, "etajima", 30, csv_export=False), VisitLog(tmp_path, "etajima"),
                       StaffConfig(password="staffpw"), themes=sel)
    app = create_app(tmp_path / "none", staff)
    auth = ("staff", "staffpw")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("192.168.1.20", 5000)),
                                 base_url="http://t") as c:
        assert (await c.get("/staff/api/theme")).status_code == 401
        st = (await c.get("/staff/api/theme", auth=auth)).json()
        assert st["override"] is None and st["wanted"] == st["auto"]
        st = (await c.post("/staff/api/theme", auth=auth, json={"name": "festival"})).json()
        assert st["override"] == "festival" and st["wanted"] == "festival"
        assert (await c.post("/staff/api/theme", auth=auth, json={"name": "nope"})).status_code == 400
        st = (await c.post("/staff/api/theme", auth=auth, json={"name": None})).json()
        assert st["override"] is None
        page = await c.get("/staff/", auth=auth)
        assert "アバターと背景" in page.text


class _ThemeEdge(_Edge):
    def __init__(self) -> None:
        super().__init__()
        self.themes: list[tuple[str, str, str]] = []
        self.credits: list[tuple[str, str]] = []

    async def send_theme(self, site: str, name: str, avatar: str, background: str,
                         credit: str = "", anchor: str = "center") -> None:
        self.themes.append((name, avatar, background))
        self.credits.append((credit, anchor))


async def test_theme_switches_only_when_nobody_is_there(tmp_path: Path, monkeypatch: Any) -> None:
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "config", replace(main_mod.config, data_dir=tmp_path, theme=ThemeConfig()))
    (tmp_path / "avatars").mkdir()
    (tmp_path / "avatars" / "idochan_office.vrm").write_bytes(b"x")
    app, _ = _loop()
    edge = _ThemeEdge()
    app._edge = edge  # type: ignore[assignment]
    assert app._camera is not None
    app._themes.site_themes = load_themes(THEMES, "etajima")
    assert not await app.check_theme()  # まだエッジがつながっていない
    app._on_edge_hello({})
    for _ in range(20):
        if edge.themes:
            break
        await main_mod.asyncio.sleep(0.01)
    auto = app._themes.applied
    assert auto is not None and edge.themes[0] == (auto.name, "avatars/idochan_office.vrm",
                                                   f"backgrounds/etajima_{auto.name}.jpg")
    # 会話中（輪がある・切り出した）は切り替えない
    app._themes.set_override("festival")
    app._camera.tracker.snapshot = PartySnapshot(PartyState.PARTY, 1, True)
    app._opener_said = True
    assert not await app.check_theme()
    app._camera.tracker.snapshot = PartySnapshot(PartyState.NONE, 0, False)
    assert not await app.check_theme()  # 輪が解けても、会話が区切られるまでは切り替えない
    await app._end_conversation()
    assert await app.check_theme()
    # 祭りのアバターのファイルが無い → 開発用のサンプルで出す（落ちない）
    assert edge.themes[-1] == ("festival", main_mod.config.edge.default_avatar, "backgrounds/etajima_autumn.jpg")
    assert not await app.check_theme()  # 同じテーマは送り直さない
    await app._tts.close()


async def test_opener_uses_applied_theme_season_line() -> None:
    import src.main as main_mod

    app, _ = _loop()
    said: list[list[str]] = []

    async def fake_respond(text: str, language: str, timing: object, source: Any = None,
                           trigger: str | None = None, speaker: str | None = None) -> None:
        said.append([s async for s in source()])

    app._respond_and_speak = fake_respond  # type: ignore[method-assign]
    app._themes.site_themes = load_themes(THEMES, "etajima")
    app._themes.applied = app._themes.site_themes.themes["festival"]
    app._weather.today = lambda now: (None, None)  # type: ignore[method-assign]
    app._say_opener()
    await main_mod.asyncio.sleep(0.05)
    assert said[0][1] == "お祭りの季節ですね。"
    if app._bye_task is not None:
        app._bye_task.cancel()
    await app._tts.close()
