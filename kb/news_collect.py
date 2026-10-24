"""お知らせの収集（WP11 A-1）。kb/build.py の「お知らせ」区画から呼ぶ。

sources の ``news.feeds``（新着・お知らせ・イベントの Atom / RSS）から記事を集め、1 記事 1 ファイルで
``$DATA/kb/<site>.news/<id>.json`` に書く。パックの版のディレクトリには ``news`` という名前でそこへのリンクを置く
（``$DATA/kb/<site>/news/<id>.json`` で読める。版を差し替えても記事と職員の変更は残る）。

- 取得は Crawler と同じ規則（許可ドメインだけ・robots.txt・同じホストへは rate_limit_sec 秒に 1 回・条件付き GET）
- 画像は記事ページの og:image か本文の最初の画像で、許可ドメイン内・記事と同じホスト・幅 image_min_width 以上の 1 枚だけ。
  長辺 image_max_edge に縮めて保存する。無ければ null（画面は題名だけの板になる）
- 本文が変わった記事（content_hash）だけ、分類・要約を夜にやり直す（職員が直した要約は上書きしない）
- 期限: 一覧の期間の終わり → 本文の期間の終わり → 取得から default_ttl_days 日。期限が過ぎて keep_days 日たったファイルは消す
- 市町のサイトが取れないとき（メンテナンス中など）は、前回取ったもの（生のキャッシュ、無ければ知識パックのキャッシュの本文）で続ける
"""

from __future__ import annotations

import email.utils
import hashlib
import io
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import httpx

if TYPE_CHECKING:
    from build import Crawler, Sources

KB_DIR = Path(__file__).resolve().parent
SERVER_DIR = KB_DIR.parent / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from src.config import NewsConfig, config
from src.news import (
    NewsRecord,
    NewsRules,
    load_records,
    load_rules,
    news_id,
    save_record,
)

logger = logging.getLogger("kb.news")

JST_OFFSET = timedelta(hours=9)
RAW_SUBDIR = "raw"


@dataclass
class FeedItem:
    url: str
    title: str = ""
    published: str = ""  # YYYY-MM-DD（一覧に無ければ空）
    start: str = ""  # 一覧の期間（RSS の event_sdate / event_edate）
    end: str = ""


@dataclass
class NewsSources:
    feeds: list[dict[str, Any]] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    label: str = ""


def news_sources(site: str) -> NewsSources | None:
    import yaml

    data = yaml.safe_load((KB_DIR / "sources" / f"{site}.yaml").read_text(encoding="utf-8"))
    raw = data.get("news")
    if not isinstance(raw, dict) or not raw.get("feeds"):
        return None
    return NewsSources(feeds=[dict(f) for f in raw["feeds"]], include=list(raw.get("include") or []),
                       exclude=list(raw.get("exclude") or []), label=str(raw.get("label") or ""))


# ----------------------------------------------------------------------
# 置き場
# ----------------------------------------------------------------------


def store_dir(root: Path, site: str) -> Path:
    """記事の実体（$DATA/kb/<site>.news）。"""
    return root / f"{site}.news"


def link_news(version: Path, store: Path, cfg: NewsConfig) -> None:
    """パックの版のディレクトリに news → 実体 のリンクを置く。"""
    store.mkdir(parents=True, exist_ok=True)
    link = version / cfg.subdir
    if link.is_symlink() and link.resolve() == store.resolve():
        return
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        logger.warning("版の中に news のディレクトリがあります（リンクにしません）: %s", link)
        return
    link.symlink_to(store, target_is_directory=True)


# ----------------------------------------------------------------------
# 取得（生のまま。フィードの日付・記事の画像に要る）
# ----------------------------------------------------------------------


def _raw_cache(crawler: Crawler, url: str) -> Path:
    d = crawler.cache_dir / RAW_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(url.encode()).hexdigest() + ".json")


def fetch_raw(crawler: Crawler, url: str, stats: dict[str, int]) -> str | None:
    """本文をそのまま取る（条件付き GET）。取れなければ前回取ったもの（あれば）。"""
    if not crawler.allowed_domain(url) or not crawler.robots_ok(url):
        return None
    cache = _raw_cache(crawler, url)
    cached: dict[str, Any] | None = None
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
    headers: dict[str, str] = {}
    if cached and cached.get("etag"):
        headers["If-None-Match"] = cached["etag"]
    if cached and cached.get("last_modified"):
        headers["If-Modified-Since"] = cached["last_modified"]
    host = urlparse(url).netloc
    response: httpx.Response | None = None
    for attempt in range(3):
        crawler._wait(host)
        try:
            response = crawler.client.get(url, headers=headers)
            if response.status_code < 500:
                break
        except httpx.HTTPError as exc:
            logger.debug("取得失敗 %s: %s", url, exc)
            response = None
        if attempt < 2:
            import time

            time.sleep(crawler.src.rate_limit_sec * (attempt + 1))
    if response is not None and response.status_code == 304 and cached:
        stats["not_modified"] += 1
        return str(cached["body"])
    if response is not None and response.status_code == 200:
        stats["fetched"] += 1
        cache.write_text(json.dumps({
            "url": url, "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"), "body": response.text,
            "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }, ensure_ascii=False), encoding="utf-8")
        return response.text
    stats["failed"] += 1
    if cached:
        stats["stale"] += 1
        logger.info("取得できないので前回の内容を使います: %s", url)
        return str(cached["body"])
    return None


# ----------------------------------------------------------------------
# 一覧（Atom / RSS 1.0 / RSS 2.0）
# ----------------------------------------------------------------------

_ENTRY_RE = re.compile(r"<entry\b[^>]*>(.*?)</entry>", re.DOTALL)
_ITEM_RE = re.compile(r"<item\b([^>]*)>(.*?)</item>", re.DOTALL)


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}\b[^>]*>(.*?)</{name}>", block, re.DOTALL)
    if not m:
        return ""
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(1), flags=re.DOTALL)
    import html

    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _day(value: str) -> str:
    """日付・日時の文字列を YYYY-MM-DD に（読めなければ空）。"""
    value = value.strip()
    if not value:
        return ""
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError:
            return ""
    try:
        dt = email.utils.parsedate_to_datetime(value)  # RSS 2.0 の pubDate
    except (TypeError, ValueError):
        return ""
    return (dt.astimezone() if dt.tzinfo else dt).date().isoformat()


def parse_feed(xml: str, base: str) -> list[FeedItem]:
    """一覧の記事（URL・題名・掲載日・期間）。"""
    items: list[FeedItem] = []
    for block in _ENTRY_RE.findall(xml):  # Atom
        href = re.search(r"<link\b(?=[^>]*rel=[\"']alternate[\"'])[^>]*href=[\"']([^\"']+)", block) \
            or re.search(r"<link\b[^>]*href=[\"']([^\"']+)", block)
        if not href:
            continue
        items.append(FeedItem(urljoin(base, href.group(1)), _tag(block, "title"),
                              _day(_tag(block, "published") or _tag(block, "updated"))))
    for attrs, block in _ITEM_RE.findall(xml):  # RSS 1.0 / 2.0
        about = re.search(r"rdf:about=\"([^\"]+)\"", attrs)
        link = _tag(block, "link") or (about.group(1) if about else "")
        if not link:
            continue
        items.append(FeedItem(urljoin(base, link), _tag(block, "title"),
                              _day(_tag(block, "dc:date") or _tag(block, "pubDate")),
                              _day(_tag(block, "nc:event_sdate")), _day(_tag(block, "nc:event_edate"))))
    return items


# ----------------------------------------------------------------------
# 期限
# ----------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(?:(?P<era>令和|R)\s*(?P<ey>元|\d{1,2})\s*年|(?P<y>\d{4})\s*年)?\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日"
    r"|(?P<sy>\d{4})/(?P<sm>\d{1,2})/(?P<sd>\d{1,2})"
)
_ZEN = str.maketrans("０１２３４５６７８９／", "0123456789/")
# 本文の日付として拾う範囲（掲載日から）。古い経緯・遠い先の計画の日付を期間に入れない
PERIOD_PAST_DAYS = 60
PERIOD_FUTURE_DAYS = 365


def body_dates(text: str, published: date) -> list[date]:
    out: list[date] = []
    for m in _DATE_RE.finditer(text.translate(_ZEN)):
        try:
            if m["sy"]:
                d = date(int(m["sy"]), int(m["sm"]), int(m["sd"]))
            else:
                month, day = int(m["m"]), int(m["d"])
                if m["era"]:
                    year = 2018 + (1 if m["ey"] == "元" else int(m["ey"]))
                elif m["y"]:
                    year = int(m["y"])
                else:  # 年の無い日付は掲載日から（掲載日より半年以上前の月なら翌年）
                    year = published.year + (1 if month < published.month - 5 else 0)
                d = date(year, month, day)
        except ValueError:
            continue
        if published - timedelta(days=PERIOD_PAST_DAYS) <= d <= published + timedelta(days=PERIOD_FUTURE_DAYS):
            out.append(d)
    return out


def find_period(item: FeedItem, title: str, body: str, published: date, rules: NewsRules) -> dict[str, str]:
    """期間（開始・終了）。一覧の期間があればそれ、無ければ本文の期間の語のある行の日付。取れなければ空。

    日付が 1 つだけで「から」「より」「～」が続く行（「10月1日から開始します」）は始まりだけ（終わりにしない）。
    """
    if item.start or item.end:
        return {k: v for k, v in (("start", item.start), ("end", item.end or item.start)) if v}
    starts: list[date] = []
    ends: list[date] = []
    for line in [title, *body.splitlines()]:
        if not any(w in line for w in rules.period_words):
            continue
        found = body_dates(line, published)
        if len(found) == 1 and _OPEN_END_RE.search(line.translate(_ZEN)):
            starts += found
        else:
            ends += found
    if not starts and not ends:
        return {}
    out = {"start": min(starts + ends).isoformat()}
    if ends:
        out["end"] = max(ends).isoformat()
    return out


# 日付の直後の「から」「より」「～」（終わりの日付が無い）
_OPEN_END_RE = re.compile(r"日\s*(?:[（(][^）)]{1,4}[）)])?\s*(?:から|より|以降|[～〜~])\s*(?![^。]*\d{1,2}\s*月\s*\d{1,2}\s*日)")


def expires_on(period: dict[str, str], fetched: date, cfg: NewsConfig) -> str:
    """候補から外す日（期間の終わりの翌日。期間が無ければ取得から default_ttl_days 日）。"""
    if period.get("end"):
        return (date.fromisoformat(period["end"]) + timedelta(days=1)).isoformat()
    return (fetched + timedelta(days=cfg.default_ttl_days)).isoformat()


# ----------------------------------------------------------------------
# 画像
# ----------------------------------------------------------------------

_OG_IMAGE_RE = re.compile(
    r"<meta\b(?=[^>]*(?:property|name)=[\"']og:image[\"'])[^>]*content=[\"']([^\"']+)[\"']", re.IGNORECASE)
_IMG_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
_BODY_START_RE = re.compile(r"<h1\b|<main\b|id=[\"'](?:main|main_body|content|contents)[\"']", re.IGNORECASE)
_BODY_END_RE = re.compile(r"<footer\b|id=[\"']footer|class=[\"'][^\"']*footer|このページの先頭へ", re.IGNORECASE)
# 本文の中でも、飾り・案内の画像（サイト共通の部品・PDF の閲覧ソフトの案内など）は記事の画像としない
_DECOR_RE = re.compile(r"/img/common/|/common/|get_adobe|adobe_reader|icon|logo|banner|btn_|button|spacer|\.svg$",
                       re.IGNORECASE)


def image_candidates(html: str, page_url: str) -> list[str]:
    """記事の画像の候補（og:image、次に本文の最初の画像）。"""
    out: list[str] = []
    og = _OG_IMAGE_RE.search(html)
    if og:
        out.append(urljoin(page_url, og.group(1).strip()))
    start = _BODY_START_RE.search(html)
    body = html[start.start():] if start else html
    end = _BODY_END_RE.search(body)
    body = body[:end.start()] if end else body
    for src in _IMG_RE.findall(body):
        url = urljoin(page_url, src.strip())
        if not _DECOR_RE.search(url) and url not in out:
            out.append(url)
            break
    return out


def image_allowed(img_url: str, page_url: str, allow_domains: list[str]) -> bool:
    """許可ドメイン内・記事と同じホスト（外部サイトの画像は取らない）。"""
    host = urlparse(img_url).hostname
    return urlparse(img_url).scheme in ("http", "https") and host in allow_domains \
        and host == urlparse(page_url).hostname


def fit_image(data: bytes, cfg: NewsConfig) -> bytes | None:
    """幅 image_min_width 未満なら None。長辺 image_max_edge に縮めた JPEG。"""
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    if img.width < cfg.image_min_width:
        return None
    img = img.convert("RGB")
    img.thumbnail((cfg.image_max_edge, cfg.image_max_edge))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=cfg.image_jpeg_quality)
    return buf.getvalue()


def fetch_image(crawler: Crawler, html: str, page_url: str, out: Path, cfg: NewsConfig) -> str | None:
    """記事の画像を 1 枚保存する。保存した元の URL（無ければ None）。"""
    for url in image_candidates(html, page_url):
        if not image_allowed(url, page_url, crawler.src.allow_domains) or not crawler.robots_ok(url):
            continue
        crawler._wait(urlparse(url).netloc)
        try:
            r = crawler.client.get(url)
        except httpx.HTTPError:
            continue
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            continue
        if not image_allowed(str(r.url), page_url, crawler.src.allow_domains):
            continue  # 転送で外部へ出た
        jpeg = fit_image(r.content, cfg)
        if jpeg is None:
            continue
        out.write_bytes(jpeg)
        return url
    return None


# ----------------------------------------------------------------------
# 収集
# ----------------------------------------------------------------------


def _cached_page(crawler: Crawler, url: str) -> dict[str, Any] | None:
    """知識パックのキャッシュ（抽出済みの本文・リンク）。生のキャッシュが無いときの最後の手段。"""
    f = crawler._cache_path(url)
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def clean_body(body: str, cfg: NewsConfig) -> str:
    """本文から、どの記事にも付く決まり文句の行（PDF の閲覧ソフトの案内など）を落とす。"""
    noise = re.compile(cfg.body_noise_pattern)
    lines = [re.sub(r"\s*\\?\[PDFファイル[^\]]*\\?\]", "", x) for x in body.splitlines() if not noise.search(x.strip())]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def content_hash(title: str, body: str) -> str:
    return hashlib.sha1(f"{title}\n{body}".encode()).hexdigest()


def collect(src: Sources, crawler: Crawler, store: Path, cfg: NewsConfig | None = None,
            now: datetime | None = None) -> dict[str, int]:
    """一覧から記事を集めて store に書く。統計を返す。"""
    import build

    cfg = cfg or config.news
    now = now or datetime.now(build.JST)
    stamp = now.isoformat(timespec="seconds")
    today = now.date()
    rules = load_rules(cfg.rules_path)
    news = news_sources(src.site)
    stats = {"items": 0, "new": 0, "changed": 0, "unchanged": 0, "excluded": 0, "no_body": 0,
             "images": 0, "fetched": 0, "not_modified": 0, "failed": 0, "stale": 0, "removed": 0}
    if news is None:
        logger.info("sources に news がありません（お知らせは集めません）")
        return stats
    store.mkdir(parents=True, exist_ok=True)
    existing = {r.id: r for r in load_records(store)}
    seen: dict[str, FeedItem] = {}
    for feed in news.feeds:
        url = str(feed["url"])
        xml = fetch_raw(crawler, url, stats)
        if xml is not None:
            items = parse_feed(xml, url)
        else:
            cached = _cached_page(crawler, url)
            items = [FeedItem(u) for u in (cached or {}).get("links", [])]
            if cached:
                stats["stale"] += 1
                logger.info("一覧を取れないので、知識パックのキャッシュのリンクを使います（掲載日なし）: %s", url)
        n = 0
        for item in items:
            if n >= cfg.max_items_per_feed:
                break
            if item.url in seen or not crawler.allowed_domain(item.url):
                continue
            if news.include and not any(re.search(p, item.url) for p in news.include):
                continue
            if any(re.search(p, item.url) for p in news.exclude):
                stats["excluded"] += 1
                continue
            seen[item.url] = item
            n += 1
    stats["items"] = len(seen)
    for url, item in seen.items():
        rid = news_id(url)
        old = existing.get(rid)
        html = fetch_raw(crawler, url, stats)
        got_on = today.isoformat()  # 取得日（一覧に掲載日が無いときの掲載日）
        if html is not None:
            title, body = build.extract_html(html, url)
        else:
            cached = _cached_page(crawler, url)
            if not cached:
                stats["no_body"] += 1
                continue
            title, body = cached.get("title", ""), cached.get("text", "")
            got_on = str(cached.get("fetched_at") or got_on)[:10]
        title = build.clean_title(item.title or title)
        page = build.Page(url, title, body, [], stamp)
        if build._excluded(page, src):
            stats["excluded"] += 1
            continue
        body = clean_body(build.redact_names(body), cfg)
        if not body:
            stats["no_body"] += 1  # 本文の無い記事（資料の一覧だけ）も残す（分類で紹介しないにする）
        chash = content_hash(title, body)
        published = item.published or (old.published if old else "") or got_on
        rec = old or NewsRecord(id=rid, title=title, url=url, published=published, fetched_at=stamp,
                                label=news.label or src.default_label)
        first_seen = date.fromisoformat(rec.fetched_at[:10]) if rec.fetched_at else today
        rec.published = published
        rec.period = find_period(item, title, body, date.fromisoformat(published[:10]), rules)
        rec.expires_at = expires_on(rec.period, first_seen, cfg)
        changed = old is None or old.content_hash != chash
        if changed:
            stats["changed" if old is not None else "new"] += 1
            rec.title, rec.body, rec.content_hash, rec.updated_at = title, body, chash, stamp
        else:
            stats["unchanged"] += 1
        # 画像は、本文が変わったとき・まだ記事のページそのものから探していないとき（サイトが止まっていて
        # キャッシュの本文で作った記事は、戻った最初の夜に探す）
        if html is not None and (changed or not rec.image_checked):
            img_path = store / f"{rid}.jpg"
            src_url = fetch_image(crawler, html, url, img_path, cfg)
            rec.image_checked = True
            if src_url is not None:
                rec.image, rec.image_src = img_path.name, src_url
                stats["images"] += 1
            elif rec.image:
                img_path.unlink(missing_ok=True)
                rec.image, rec.image_src = None, None
        save_record(store, rec)
    stats["removed"] = prune(store, today, cfg)
    logger.info("お知らせの収集: %s", stats)
    return stats


def prune(store: Path, today: date, cfg: NewsConfig) -> int:
    """期限が過ぎて、取得から keep_days 日たった記事（と画像）を消す。"""
    removed = 0
    for r in load_records(store):
        first = date.fromisoformat(r.fetched_at[:10]) if r.fetched_at else today
        if r.expired(today) and (today - first).days >= cfg.keep_days:
            (store / f"{r.id}.json").unlink(missing_ok=True)
            (store / f"{r.id}.jpg").unlink(missing_ok=True)
            removed += 1
    return removed
