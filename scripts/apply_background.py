# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pillow>=10"]
# ///
"""人間が選んだ Commons の写真を背景にする（WP10）。

    uv run scripts/apply_background.py --site akiota --season autumn --no 1
    uv run scripts/apply_background.py --site akiota --season winter --file "File:Osorakan ski resort.jpg"
    uv run scripts/apply_background.py --site etajima --season winter --no 9 --anchor right
    uv run scripts/apply_background.py --list          # 適用済みの一覧（credits.json の内容）

- --no は ``~/idobata-data/backgrounds/candidates/<site>/candidates.json`` の番号、--file は Commons のファイル名
- どちらでも Commons API でメタデータ（撮影者・ライセンス・説明・日付）を取り直し、CC0・CC BY・CC BY-SA（4.0／3.0）・
  パブリックドメインだけ受け付ける。--no のときは候補を集めたときのライセンスと食い違えば止まる
- 元画像（upload.wikimedia.org の原寸）を取り、幅 1920 に合わせて縮小し、高さの余りを上 40%・下 60% で切って 1920×1080 にする
  （空を残して手前を切る）。下を少し暗くする（アバターの白いシャツが溶けないように、字幕を読みやすく）
- 縦置きの画面ではエッジが横 1/3 弱だけを見せる。その位置を --anchor left|center|right で決め、credits.json に残す
  （エッジは background-position の横位置に使う）。確かめ用のプレビューを ``backgrounds/previews/`` に出す
- 出力は ``~/idobata-data/backgrounds/<site>_<season>.jpg``（git に入れない）と ``credits.json``

Commons への要求は fetch_commons_backgrounds.py と同じ User-Agent で、間を 3 秒以上空ける。429 なら 30 秒待って 1 回だけやり直す。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_commons_backgrounds import API, LICENSE_URLS, LICENSES, USER_AGENT, plain

DATA = Path(os.environ.get("IDOBATA_DATA_DIR", str(Path.home() / "idobata-data")))
BG_DIR = DATA / "backgrounds"
CREDITS = BG_DIR / "credits.json"
JST = timezone(timedelta(hours=9))
SEASONS = ("spring", "summer", "autumn", "winter")
OUT_W, OUT_H = 1920, 1080
TOP_SHARE = 0.4  # 高さの余りのうち上で切る割合（残りは下で切る）
JPEG_QUALITY = 90
# 下を暗くする（下から DARKEN_HEIGHT の範囲で、いちばん下を DARKEN_MAX だけ暗く）。アバターの白いシャツ
# （横置きで画面の下 55%、縦置きで下 60%）が明るい空・海・雪に溶けないように、字幕も読みやすく
DARKEN_HEIGHT = 0.6
DARKEN_MAX = 0.45
DARKEN_CURVE = 1.2
REQUEST_GAP_SEC = 3.0
RETRY_429_SEC = 30.0
PD_LICENSE = "パブリックドメイン"


class CommonsError(RuntimeError):
    pass


class Commons:
    """要求の間を 3 秒以上空け、429 なら 30 秒待って 1 回だけやり直す。"""

    def __init__(self) -> None:
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120)
        self._last = 0.0

    def get(self, url: str, **params: Any) -> httpx.Response:
        for attempt in range(2):
            wait = self._last + REQUEST_GAP_SEC - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.client.get(url, params=params or None)
            except httpx.HTTPError as exc:
                self._last = time.monotonic()
                if attempt == 0:
                    time.sleep(RETRY_429_SEC)
                    continue
                raise CommonsError(f"取得に失敗しました: {url}: {exc}") from exc
            self._last = time.monotonic()
            if r.status_code == 429 and attempt == 0:
                print(f"429 が返りました。{RETRY_429_SEC:.0f} 秒待って 1 回だけやり直します", file=sys.stderr)
                time.sleep(RETRY_429_SEC)
                continue
            if r.status_code != 200:
                raise CommonsError(f"HTTP {r.status_code}: {url}")
            return r
        raise CommonsError(f"2 回続けて失敗しました: {url}")

    def info(self, title: str) -> dict[str, Any]:
        r = self.get(API, action="query", titles=title, prop="imageinfo", iiprop="url|size|extmetadata|mime",
                     format="json", formatversion="2")
        pages = r.json().get("query", {}).get("pages", [])
        if not pages or "imageinfo" not in pages[0]:
            raise CommonsError(f"Commons にありません: {title}")
        return dict(pages[0])


def license_of(meta: dict[str, Any]) -> tuple[str, bool] | None:
    """(ライセンス, クレジットが要るか)。受け付けないものは None。"""
    raw = plain(meta.get("LicenseShortName", {}).get("value", ""))
    short = raw.lower().replace("cc-by-sa-", "cc by-sa ").replace("cc-by-", "cc by ")
    if short in LICENSES:
        lic = LICENSES[short]
        return lic, lic != "CC0"
    if short.startswith(("public domain", "pd")) or meta.get("License", {}).get("value", "").lower().startswith("pd"):
        return PD_LICENSE, False
    return None


def fit(img: Image.Image) -> tuple[Image.Image, str]:
    """1920×1080 に合わせる。(画像, 切り出し方の説明)。"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if w * OUT_H >= h * OUT_W:  # 16:9 より横長: 高さに合わせて横を中央で切る
        nw = round(w * OUT_H / h)
        img = img.resize((nw, OUT_H), Image.Resampling.LANCZOS)
        left = (nw - OUT_W) // 2
        return img.crop((left, 0, left + OUT_W, OUT_H)), f"高さ 1080 に縮小し、左右 {nw - OUT_W}px を中央で切る"
    nh = round(h * OUT_W / w)
    img = img.resize((OUT_W, nh), Image.Resampling.LANCZOS)
    extra = nh - OUT_H
    top = round(extra * TOP_SHARE)
    scale = "縮小" if w >= OUT_W else "拡大"
    return img.crop((0, top, OUT_W, top + OUT_H)), f"幅 1920 に{scale}し、上 {top}px・下 {extra - top}px を切る"


def darken_bottom(img: Image.Image) -> Image.Image:
    band = int(OUT_H * DARKEN_HEIGHT)
    mask = Image.new("L", (OUT_W, OUT_H), 0)
    draw = ImageDraw.Draw(mask)
    for i in range(band):
        draw.line([(0, OUT_H - band + i), (OUT_W, OUT_H - band + i)], fill=int(255 * DARKEN_MAX * (i / band) ** DARKEN_CURVE))
    return Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask)


def portrait_window(anchor: str) -> tuple[int, int]:
    """縦置き 1080×1920 で cover したときに見える横の範囲（1920×1080 の画像の座標）。"""
    visible = round(OUT_H * 1080 / 1920)  # 縦 1920 に合わせると 1920×1080 → 3413×1920。見えるのは 1080/3413
    left = {"left": 0, "center": (OUT_W - visible) // 2, "right": OUT_W - visible}[anchor]
    return left, left + visible


def preview(img: Image.Image, path: Path) -> None:
    """横の全体に、縦置きで見える範囲（左・中央・右）を重ねた確かめ用の画像。"""
    p = img.copy()
    d = ImageDraw.Draw(p)
    for anchor, color in (("left", (255, 80, 80)), ("center", (255, 255, 0)), ("right", (80, 160, 255))):
        a, b = portrait_window(anchor)
        d.rectangle((a + 2, 2, b - 2, OUT_H - 3), outline=color, width=6)
        d.text((a + 12, 12), anchor, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    p.resize((960, 540)).save(path, quality=85)


def load_credits() -> dict[str, Any]:
    if CREDITS.is_file():
        return dict(json.loads(CREDITS.read_text(encoding="utf-8")))
    return {}


def candidate(site: str, no: int) -> dict[str, Any]:
    path = BG_DIR / "candidates" / site / "candidates.json"
    for c in json.loads(path.read_text(encoding="utf-8"))["candidates"]:
        if c["no"] == no:
            return dict(c)
    raise SystemExit(f"{path} に No.{no} がありません")


def apply(site: str, season: str, title: str, anchor: str, expected_license: str | None) -> dict[str, Any]:
    commons = Commons()
    page = commons.info(title)
    info = page["imageinfo"][0]
    meta = info.get("extmetadata") or {}
    lic = license_of(meta)
    if lic is None:
        raise SystemExit(f"受け付けないライセンスです: {meta.get('LicenseShortName', {}).get('value')}（{title}）")
    license_name, credit_required = lic
    if expected_license and expected_license != license_name:
        raise SystemExit(f"ライセンスが候補の記録（{expected_license}）と API（{license_name}）で食い違います: {title}")
    original = str(info["url"])
    if not original.startswith("https://upload.wikimedia.org/") or "/thumb/" in original:
        raise SystemExit(f"原寸の URL ではありません: {original}")
    # 原寸は手元に取っておく（切り出しを直すたびに Commons から取り直さない。git の外）
    cache = BG_DIR / ".originals" / original.split("?")[0].rsplit("/", 1)[-1]
    if cache.is_file():
        raw = cache.read_bytes()
    else:
        raw = commons.get(original).content
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    img, crop = fit(Image.open(io.BytesIO(raw)))
    img = darken_bottom(img)
    BG_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{site}_{season}.jpg"
    img.save(BG_DIR / name, quality=JPEG_QUALITY, optimize=True)
    preview(img, BG_DIR / "previews" / name)
    author = plain(meta.get("Artist", {}).get("value", "")) or "（不明）"
    entry = {
        "file": name,
        "site": site,
        "season": season,
        "commons_title": page["title"],
        "commons_url": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{page['title'].replace(' ', '_')}",
        "original_url": original.split("?")[0],
        "author": author,
        "license": license_name,
        "license_url": LICENSE_URLS.get(license_name) or plain(meta.get("LicenseUrl", {}).get("value", "")),
        "credit_required": credit_required,
        "date": plain(meta.get("DateTimeOriginal", {}).get("value", ""))[:20],
        "description": plain(meta.get("ImageDescription", {}).get("value", ""))[:200],
        "original_size": [int(info["width"]), int(info["height"])],
        "crop": crop + f"。下 {DARKEN_HEIGHT:.0%} を最大 {DARKEN_MAX:.0%} 暗くする",
        "anchor": anchor,
        "applied_at": datetime.now(JST).isoformat(timespec="seconds"),
    }
    credits = load_credits()
    credits[f"{site}_{season}"] = entry
    tmp = CREDITS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(credits, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, CREDITS)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["akiota", "etajima"])
    ap.add_argument("--season", choices=SEASONS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no", type=int, help="candidates.json の番号")
    g.add_argument("--file", help='Commons のファイル名（"File:..."）')
    ap.add_argument("--anchor", choices=["left", "center", "right"], default="center",
                    help="縦置きの画面で見せる横の位置")
    ap.add_argument("--list", action="store_true", help="適用済みの一覧")
    args = ap.parse_args()
    if args.list:
        for key, e in sorted(load_credits().items()):
            print(f"{key:<16} {e['license']:<12} {e['anchor']:<6} {e['author'][:24]:<24} {e['commons_title']}"
                  f"{'' if e['credit_required'] else '（クレジット不要）'}")
        return 0
    if not (args.site and args.season and (args.no or args.file)):
        ap.error("--site・--season と、--no か --file が要ります")
    expected = None
    title = args.file
    if args.no:
        c = candidate(args.site, args.no)
        title, expected = c["title"], c["license"]
    if not title.startswith("File:"):
        title = "File:" + title
    try:
        e = apply(args.site, args.season, title, args.anchor, expected)
    except CommonsError as exc:
        print(f"止まりました: {exc}", file=sys.stderr)
        return 1
    print(f"{e['file']}: {e['commons_title']} / {e['author']} / {e['license']} / {e['crop']} / anchor={e['anchor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
