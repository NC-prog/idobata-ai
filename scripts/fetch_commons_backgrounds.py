# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pillow>=10"]
# ///
"""背景の実写の候補を Wikimedia Commons から集める（WP10）。選ぶのは人間。

    uv run scripts/fetch_commons_backgrounds.py [--site akiota --site etajima] [--max 40]

- Commons の API（https://commons.wikimedia.org/w/api.php）で、拠点ごとの検索語でファイル（名前空間 6）を検索する
- ライセンスが CC0・CC BY・CC BY-SA（4.0／3.0）のものだけ、横長（幅 1600px 以上、幅＞高さ）に絞る
- 各候補の元画像の URL・撮影者・ライセンス・Commons のページ URL を
  ``~/idobata-data/backgrounds/candidates/<site>/candidates.json`` に、縮小画像（幅 480）を同じ場所に ``NN.jpg`` で保存する
- 季節（紅葉・雪・桜・新緑・夏）はファイル名・説明・カテゴリの語から推定してタグを付ける（推定なので人が確かめる）
- 一覧シート（番号・撮影者・ライセンス・推定季節を印字した PNG）を ``docs/img/bg_candidates_<site>.png`` に出す

外へ出るのは Commons への検索と画像の取得だけ（会話・音声・映像とは無関係）。Wikimedia の User-Agent の方針に従って名乗る。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "idobata-ai-backgrounds/0.1 (+Fuji Seisakusho LLC; municipal lobby guide pilot; nc.tokyo2006@gmail.com)"
REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("IDOBATA_DATA_DIR", str(Path.home() / "idobata-data")))

QUERIES: dict[str, list[str]] = {
    "akiota": ["三段峡", "Sandankyo", "温井ダム", "Nukui Dam", "深入山", "龍頭峡", "安芸太田町 紅葉", "安芸太田町 雪",
               "Akiota", "安芸太田町", "三段峡 紅葉", "Sandankyo autumn", "恐羅漢山", "Mount Osorakan", "加計 安芸太田",
               "戸河内", "筒賀", "Ryuzukyo", "Shinnyuzan", "太田川 安芸太田"],
    "etajima": ["江田島", "Etajima", "能美島", "古鷹山", "牡蠣筏", "Oyster rafts Hiroshima", "江田島湾", "早瀬大橋",
                "Etajima island", "Nomijima", "沖美町 江田島", "大柿町 江田島", "Mount Furutaka", "Etajima Hiroshima",
                "江田島 海", "能美町", "江田島市", "Etajima panoramio", "Okimi Hiroshima", "Ogaki Etajima", "Nomi Etajima",
                "Etajima Hiroshima Prefecture"],
}
CATEGORY_SKIP = ["building", "transport", "map", "symbol", "naval", "academy", "imo ", "people", "you me", "shop",
                 "train", "vehicle", "logo", "flag", "seal", "museum", "sign", "interior"]
# Commons のカテゴリ（拠点の写真の束。下のカテゴリもたどる。乗り物・建物・地図などの枝は EXCLUDE_WORDS で外す）
CATEGORIES: dict[str, list[str]] = {
    "akiota": ["Category:Akiota, Hiroshima", "Category:Sandankyo", "Category:Ryuzukyo", "Category:Mount Shinnyu"],
    "etajima": ["Category:Etajima, Hiroshima", "Category:Etajima Bay"],
}
CATEGORY_DEPTH = 3
# その拠点の写真であることの手がかり（ファイル名・説明・カテゴリのどれかに要る。同名の別の土地を外す）
SITE_WORDS: dict[str, list[str]] = {
    "akiota": ["安芸太田", "akiota", "三段峡", "sandankyo", "sandandaki", "mitsudaki", "温井", "nukui", "深入山",
               "shinnyu", "龍頭", "ryuzu", "筒賀", "tsutsuga", "加計", "kake", "戸河内", "togouchi", "十方山", "jippo",
               "恐羅漢", "osorakan", "kyurakan", "龍姫湖", "滝山峡", "takiyamakyo", "吉ヶ原", "yoshigahara", "黒淵",
               "三段滝", "二段滝"],
    "etajima": ["江田島", "etajima", "能美", "nomijima", "nōmi", "沖美", "大柿", "古鷹", "furutaka", "早瀬", "hayase", "hajase",
                "小用", "kumihama", "牡蠣"],
}
# 背景に向かない被写体（乗り物・建物の正面・制服・艦艇など）。ファイル名とカテゴリで外す
EXCLUDE_WORDS = ["bus", "バス", "station", "駅", "school", "学校", "post office", "郵便局", "fire station", "消防",
                 "bank", "銀行", "uniform", "jmsdf", "js ", "mso", "kiha", "rosenbus", "tailor", "club", "stadium",
                 "city hall", "town hall", "power station", "発電所", "記念碑", "三角点", "interchange", "e2a", " pa.",
                 "ship", "gifu", "岐阜", "kagami-no", "seamax", "ferry", "hotel", "temple", "寺", "jaf", "road ", "route", "r191", "kisen", "hospital", "病院", " hq", "厳島", "itsukushima", "miyajima"]
# ライセンスの短い名前（extmetadata の LicenseShortName）。これに当たるものだけ使う
LICENSES = {
    "cc0": "CC0", "cc-zero": "CC0", "cc0 1.0": "CC0",
    "cc by 4.0": "CC BY 4.0", "cc by 3.0": "CC BY 3.0",
    "cc by-sa 4.0": "CC BY-SA 4.0", "cc by-sa 3.0": "CC BY-SA 3.0",
}
LICENSE_URLS = {
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/deed.ja",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/deed.ja",
    "CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/deed.ja",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/deed.ja",
    "CC BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/deed.ja",
}
MIN_WIDTH = 1600
THUMB_WIDTH = 480
# 季節の推定（ファイル名・説明・カテゴリに含まれる語）
SEASON_WORDS: dict[str, list[str]] = {
    "紅葉": ["紅葉", "もみじ", "autumn", "fall", "momiji", "autumn leaves", "foliage", "秋"],
    "雪": ["雪", "snow", "winter", "冬", "樹氷", "ski"],
    "桜": ["桜", "さくら", "sakura", "cherry blossom", "cherry"],
    "新緑": ["新緑", "fresh green", "spring", "春", "若葉"],
    "夏": ["夏", "summer", "海水浴", "beach"],
}


def api(client: httpx.Client, **params: Any) -> dict[str, Any]:
    """Commons の API（失敗したら 3 回まで待って取り直す）。"""
    params = {"format": "json", "formatversion": "2", **params}
    for attempt in range(3):
        try:
            r = client.get(API, params=params, timeout=30)
            r.raise_for_status()
            return dict(r.json())
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def plain(text: str) -> str:
    """extmetadata の HTML（撮影者・説明）を文字だけにする。"""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


def search_titles(client: httpx.Client, query: str, limit: int = 50) -> list[str]:
    data = api(client, action="query", list="search", srsearch=f"{query} filetype:bitmap", srnamespace=6,
               srlimit=limit)
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def image_infos(client: httpx.Client, titles: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(titles), 40):
        data = api(client, action="query", titles="|".join(titles[i:i + 40]), prop="imageinfo|categories",
                   iiprop="url|size|extmetadata|mime", iiurlwidth=THUMB_WIDTH, cllimit="max", clshow="!hidden")
        out += data.get("query", {}).get("pages", [])
        time.sleep(0.5)
    return out


def license_of(meta: dict[str, Any]) -> str | None:
    short = plain(meta.get("LicenseShortName", {}).get("value", "")).lower()
    short = short.replace("cc-by-sa-", "cc by-sa ").replace("cc-by-", "cc by ")
    return LICENSES.get(short)


def seasons_of(text: str) -> list[str]:
    t = text.lower()
    return [s for s, words in SEASON_WORDS.items() if any(w.lower() in t for w in words)]


def category_files(client: httpx.Client, roots: list[str]) -> list[tuple[str, str]]:
    """カテゴリとその下（CATEGORY_DEPTH 段まで）のファイル。(ファイル名, 見つけたカテゴリ)。"""
    out: list[tuple[str, str]] = []
    seen_cats: set[str] = set()
    queue = [(c, 0) for c in roots]
    while queue:
        cat, depth = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont: dict[str, Any] = {}
        while True:
            data = api(client, action="query", list="categorymembers", cmtitle=cat, cmtype="file|subcat",
                       cmlimit=500, **cont)
            for m in data.get("query", {}).get("categorymembers", []):
                if m["ns"] == 6:
                    out.append((m["title"], cat.removeprefix("Category:")))
                elif m["ns"] == 14 and depth < CATEGORY_DEPTH and not any(w in m["title"].lower() for w in EXCLUDE_WORDS
                                                                           + CATEGORY_SKIP):
                    queue.append((m["title"], depth + 1))
            if "continue" not in data:
                break
            cont = {"cmcontinue": data["continue"]["cmcontinue"]}
        time.sleep(0.3)
    return out


def collect(client: httpx.Client, site: str, limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    titles: list[tuple[str, str]] = []
    in_category: set[str] = set()
    for t, cat in category_files(client, CATEGORIES[site]):
        if t not in seen:
            seen.add(t)
            in_category.add(t)
            titles.append((t, cat))
    for q in QUERIES[site]:
        for t in search_titles(client, q):
            if t not in seen:
                seen.add(t)
                titles.append((t, q))
        time.sleep(0.5)
    query_of = dict(titles)
    cands: list[dict[str, Any]] = []
    for page in image_infos(client, [t for t, _ in titles]):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        lic = license_of(meta)
        if lic is None or w < MIN_WIDTH or w <= h or info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        cats = [c["title"].removeprefix("Category:") for c in page.get("categories") or []]
        desc = plain(meta.get("ImageDescription", {}).get("value", ""))
        title = page["title"]
        subject = " ".join([title, *cats]).lower()
        if any(w in subject for w in EXCLUDE_WORDS):
            continue
        if title not in in_category and not any(w in f"{subject} {desc.lower()}" for w in SITE_WORDS[site]):
            continue
        cands.append({
            "title": title,
            "query": query_of.get(title, ""),
            "author": plain(meta.get("Artist", {}).get("value", "")) or "（不明）",
            "license": lic,
            "license_url": LICENSE_URLS[lic],
            "attribution_required": plain(meta.get("AttributionRequired", {}).get("value", "")),
            "credit": plain(meta.get("Credit", {}).get("value", "")),
            "description": desc[:300],
            "categories": cats,
            "width": w, "height": h,
            "original_url": info.get("url"),
            "thumb_url": info.get("thumburl"),
            "page_url": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            "seasons": seasons_of(" ".join([title, desc, *cats])),
            "date": plain(meta.get("DateTimeOriginal", {}).get("value", ""))[:20],
        })
    # 季節の手がかりがあるもの・解像度の高いものを先に（同じ点数なら検索の順）
    cands.sort(key=lambda c: (-len(c["seasons"]), -c["width"]))
    return cands[:limit]


def save_thumbs(client: httpx.Client, site: str, cands: list[dict[str, Any]]) -> Path:
    d = DATA / "backgrounds" / "candidates" / site
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.jpg"):
        old.unlink()
    for i, c in enumerate(cands, 1):
        c["no"] = i
        c["thumb_file"] = f"{i:02d}.jpg"
        for attempt in range(3):
            try:
                r = client.get(c["thumb_url"], timeout=60)
                r.raise_for_status()
                (d / c["thumb_file"]).write_bytes(r.content)
                break
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        time.sleep(0.3)
    (d / "candidates.json").write_text(json.dumps({"site": site, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                   "candidates": cands}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return d


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(f).is_file():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def contact_sheet(site: str, d: Path, cands: list[dict[str, Any]]) -> Path:
    """番号・撮影者・ライセンス・推定季節を印字した一覧シート。"""
    cols, tw, th, pad, cap = 5, 360, 203, 14, 70
    rows = (len(cands) + cols - 1) // cols
    head = 70
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, head + rows * (th + cap + pad) + pad), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    name = {"akiota": "安芸太田", "etajima": "江田島"}.get(site, site)
    draw.text((pad, 18), f"背景の候補（{name}・Wikimedia Commons・{len(cands)} 枚）  番号で選んで scripts/apply_background.py に渡す",
              fill=(20, 20, 20), font=font(26))
    f_no, f_txt = font(22), font(15)
    for i, c in enumerate(cands):
        x = pad + (i % cols) * (tw + pad)
        y = head + (i // cols) * (th + cap + pad)
        img = Image.open(d / c["thumb_file"]).convert("RGB")
        img.thumbnail((tw, th))
        sheet.paste(img, (x + (tw - img.width) // 2, y + (th - img.height) // 2))
        draw.rectangle((x, y, x + 44, y + 30), fill=(0, 0, 0))
        draw.text((x + 6, y + 2), f"{c['no']:02d}", fill=(255, 255, 255), font=f_no)
        author = c["author"] if len(c["author"]) <= 26 else c["author"][:25] + "…"
        season = "・".join(c["seasons"]) or "季節不明"
        draw.text((x, y + th + 4), f"{author}", fill=(30, 30, 30), font=f_txt)
        draw.text((x, y + th + 24), f"{c['license']}　{c['width']}×{c['height']}", fill=(60, 60, 60), font=f_txt)
        draw.text((x, y + th + 44), f"推定: {season}　{c['query']}", fill=(150, 60, 0) if c["seasons"] else (110, 110, 110),
                  font=f_txt)
    out = REPO / "docs" / "img" / f"bg_candidates_{site}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", action="append", choices=sorted(QUERIES), help="拠点（複数可。省略で両方）")
    ap.add_argument("--max", type=int, default=40, help="拠点ごとの候補の上限（20〜40）")
    args = ap.parse_args()
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for site in args.site or sorted(QUERIES):
            cands = collect(client, site, args.max)
            if len(cands) < 20:
                print(f"{site}: 候補が {len(cands)} 枚しかありません（20 枚に届かない）", file=sys.stderr)
            d = save_thumbs(client, site, cands)
            sheet = contact_sheet(site, d, cands)
            print(f"{site}: 候補 {len(cands)} 枚 → {d}/candidates.json, {sheet}")
            for c in cands:
                print(f"  {c['no']:02d} {c['license']:<12} {c['width']}x{c['height']} {'・'.join(c['seasons']) or '-':<6} "
                      f"{c['author'][:30]} | {c['title'][5:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
