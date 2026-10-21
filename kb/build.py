"""知識パックのビルド: 取得 → 抽出 → チャンク → 埋め込み → 保存（WP03）。

ネットにつながる機体で夜間に回し、成果物（1 ディレクトリ）をフロントに配る。
7560 の standalone では同じ機体で作って、そのまま使う。

    uv run --project kb python kb/build.py --site etajima          # 全部
    uv run --project kb python kb/build.py --site etajima --weather-only   # 天気だけ（朝のバッチ）

出力: ``$IDOBATA_DATA_DIR/kb/<site>/``（既定 ``~/idobata-data/kb/<site>/``）
    manifest.json / chunks.parquet / index.lancedb/ / weather.json
``kb/pack/<site>`` はそこへのシンボリックリンク（git 管理外）。

取得は robots.txt を守り、同じドメインには rate_limit_sec 秒に 1 回まで。前回の
ETag / Last-Modified を覚えておき、変わっていないページは本文を取り直さない（キャッシュは
``$IDOBATA_DATA_DIR/kb/cache/<site>/``）。個人名を含みうるページ（議事録・入札・人事など）は
sources の exclude / exclude_title_keywords で外す。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import yaml

logger = logging.getLogger("kb.build")

KB_DIR = Path(__file__).resolve().parent
REPO_ROOT = KB_DIR.parent
JST = timezone(timedelta(hours=9))

# 埋め込み（server/src/knowledge.py は manifest の値を読んで同じものを使う）
EMBED_MODEL = "cl-nagoya/ruri-v3-30m"
QUERY_PREFIX = "検索クエリ: "  # Ruri v3 のモデルカードどおり
DOC_PREFIX = "検索文書: "
EMBED_BATCH = 32

# チャンク（見出し単位を基本に、長い節は CHUNK_MAX 字で区切って CHUNK_OVERLAP 字重ねる）
CHUNK_MIN = 400
CHUNK_MAX = 800
CHUNK_OVERLAP = 100
CHUNK_DROP_BELOW = 40  # これより短い断片は捨てる（ナビの残りなど）

LANCE_TABLE = "chunks"
HTTP_TIMEOUT_SEC = 20.0
RENDER_TIMEOUT_MS = 30000
SITEMAP_MAX_DEPTH = 2
HTTP_RETRIES = 2


def data_dir() -> Path:
    """IDOBATA_DATA_DIR（deploy/standalone/.env の値も見る）。"""
    env = os.environ.get("IDOBATA_DATA_DIR")
    if not env:
        env_file = REPO_ROOT / "deploy" / "standalone" / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("IDOBATA_DATA_DIR="):
                    env = os.path.expandvars(line.split("=", 1)[1].strip().strip("'\""))
    return Path(env) if env else Path.home() / "idobata-data"


# ----------------------------------------------------------------------
# ソース定義
# ----------------------------------------------------------------------


@dataclass
class Seed:
    """起点。include / exclude は URL への正規表現（re.search）。

    scope_breadcrumbs を書くと、パンくずにそのどれかを含むページだけを採用し、
    そこからだけリンクを辿る（どのページにもある全分類へのナビを辿らないため）。
    """

    url: str
    depth: int = 1
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    scope_breadcrumbs: list[str] = field(default_factory=list)
    # url が sitemap.xml（索引なら子も読む）。載っている URL を include で絞って採用候補にする
    sitemap: bool = False
    # "playwright": ブラウザで描画してから本文を取る（JS で描画するサイト）
    render: str = ""
    # 本文にこのどれかを含むページだけ採用する（県の観光サイトから拠点の分だけ取る、など）
    require_text: list[str] = field(default_factory=list)
    # この seed で取得する上限（大きなサイト用）
    max_fetches: int = 0
    # キャッシュがこの日数より新しければ取り直さない（大きなサイトを数日かけて回す）
    recheck_days: float = 0
    # 出どころの信頼度（official / semi / news）と、応答で出典を言うときの言い方
    trust: str = ""
    label: str = ""

    def matches(self, patterns: list[str], url: str) -> bool:
        return any(re.search(p, url) for p in patterns)

    def in_scope(self, page: Page) -> bool:
        if not self.scope_breadcrumbs:
            return True
        return any(k in b for b in page.breadcrumbs for k in self.scope_breadcrumbs)


@dataclass
class Sources:
    site: str
    allow_domains: list[str]
    seeds: list[Seed]
    exclude_title_keywords: list[str]
    exclude_breadcrumb_keywords: list[str]
    default_trust: str
    default_label: str
    weather: dict[str, Any]
    rate_limit_sec: float
    user_agent: str
    respect_robots: bool
    max_pages: int
    max_fetches: int


def load_sources(site: str) -> Sources:
    data = yaml.safe_load((KB_DIR / "sources" / f"{site}.yaml").read_text(encoding="utf-8"))
    crawl = data.get("crawl", {})
    return Sources(
        site=data["site"],
        allow_domains=list(data["allow_domains"]),
        seeds=[Seed(**s) for s in data["seeds"]],
        exclude_title_keywords=list(data.get("exclude_title_keywords", [])),
        exclude_breadcrumb_keywords=list(data.get("exclude_breadcrumb_keywords", [])),
        default_trust=str(data.get("default_trust", "official")),
        default_label=str(data.get("default_label", "")),
        weather=dict(data.get("weather", {})),
        rate_limit_sec=float(crawl.get("rate_limit_sec", 1.0)),
        user_agent=str(crawl.get("user_agent", "idobata-ai-kb/0.1")),
        respect_robots=bool(crawl.get("respect_robots", True)),
        max_pages=int(crawl.get("max_pages", 500)),
        max_fetches=int(crawl.get("max_fetches", 1500)),
    )


# ----------------------------------------------------------------------
# 取得
# ----------------------------------------------------------------------


@dataclass
class Page:
    url: str
    title: str
    text: str  # 抽出した本文（Markdown。見出しは "#"）
    links: list[str]
    fetched_at: str
    # パンくず（「トップページ > 組織で情報を探す > 議会事務局 > …」）。分類での除外に使う
    breadcrumbs: list[str] = field(default_factory=list)
    trust: str = ""
    label: str = ""


class Crawler:
    """robots.txt・レート制限・条件付き GET（ETag / Last-Modified）付きの小さなクローラ。"""

    def __init__(self, src: Sources, cache_dir: Path) -> None:
        self.src = src
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            headers={"User-Agent": src.user_agent}, timeout=HTTP_TIMEOUT_SEC, follow_redirects=True,
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        # sitemap に載っていた URL の最終更新（キャッシュより古ければ取り直さない）
        self.lastmod: dict[str, datetime] = {}
        self._browser: Any = None
        self._playwright: Any = None
        self._last_request: dict[str, float] = {}
        self.stats = {"fetched": 0, "not_modified": 0, "failed": 0, "robots_blocked": 0,
                      "cache_fresh": 0, "rendered": 0}

    def close(self) -> None:
        self.client.close()
        if self._browser is not None:
            self._browser.close()
            self._playwright.stop()

    def _render(self, url: str) -> tuple[int, str, str]:
        """ブラウザ（Playwright の Chromium、sudo 不要で導入）で描画した HTML。"""
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch()
        page = self._browser.new_page(user_agent=self.src.user_agent)
        try:
            resp = page.goto(url, wait_until="networkidle", timeout=RENDER_TIMEOUT_MS)
            html = page.content()
            return (resp.status if resp else 0), html, page.url
        finally:
            page.close()

    def allowed_domain(self, url: str) -> bool:
        return urlparse(url).hostname in self.src.allow_domains

    def _wait(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            delay = self.src.rate_limit_sec - (time.monotonic() - last)
            if delay > 0:
                time.sleep(delay)
        self._last_request[host] = time.monotonic()

    def robots_ok(self, url: str) -> bool:
        if not self.src.respect_robots:
            return True
        parts = urlparse(url)
        host = parts.netloc
        if host not in self._robots:
            robots_url = f"{parts.scheme}://{host}/robots.txt"
            self._wait(host)
            try:
                r = self.client.get(robots_url)
            except httpx.HTTPError:
                r = None
            if r is not None and r.status_code == 200 and "text/plain" in r.headers.get(
                "content-type", ""
            ):
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(r.text.splitlines())
                self._robots[host] = rp
                logger.info("robots.txt あり: %s", robots_url)
            else:
                self._robots[host] = None  # 無い（404 など）なら制限なし
                logger.info("robots.txt なし（制限なしとして扱う）: %s", robots_url)
        rp = self._robots[host]
        return rp is None or rp.can_fetch(self.src.user_agent, url)

    def _cache_is_fresh(self, url: str, cached: dict[str, Any], recheck_days: float) -> bool:
        """sitemap の lastmod がキャッシュより古い、または recheck_days 以内に取ったなら取り直さない。"""
        try:
            fetched = datetime.fromisoformat(cached["fetched_at"])
        except (KeyError, ValueError):
            return False
        mod = self.lastmod.get(url)
        if mod is not None and mod <= fetched:
            return True
        return recheck_days > 0 and datetime.now(JST) - fetched < timedelta(days=recheck_days)

    def _fetch_rendered(self, url: str, cache_file: Path) -> Page | None:
        host = urlparse(url).netloc
        self._wait(host)
        try:
            status, html, final_url = self._render(url)
        except Exception:
            logger.debug("描画に失敗: %s", url, exc_info=True)
            self.stats["failed"] += 1
            return None
        if status != 200:
            self.stats["failed"] += 1
            return None
        title, text = extract_html(html, url)
        page = Page(url, title, text, extract_links(html, final_url),
                    datetime.now(JST).isoformat(timespec="seconds"), extract_breadcrumbs(html))
        self.stats["rendered"] += 1
        self._save_cache(cache_file, url, page, None, None)
        return page

    def _save_cache(self, cache_file: Path, url: str, page: Page,
                    etag: str | None, last_modified: str | None) -> None:
        cache_file.write_text(json.dumps({
            "url": url, "etag": etag, "last_modified": last_modified,
            "title": page.title, "text": page.text, "links": page.links,
            "breadcrumbs": page.breadcrumbs, "fetched_at": page.fetched_at,
        }, ensure_ascii=False), encoding="utf-8")

    def sitemap_urls(self, url: str, depth: int = 0) -> list[str]:
        """sitemap.xml の URL（索引なら子の sitemap も読む）。lastmod は self.lastmod に覚える。"""
        if depth > SITEMAP_MAX_DEPTH or not self.robots_ok(url):
            return []
        self._wait(urlparse(url).netloc)
        try:
            r = self.client.get(url)
            r.raise_for_status()
        except httpx.HTTPError:
            logger.warning("sitemap を取得できません: %s", url)
            return []
        out: list[str] = []
        for block in re.findall(r"<(?:url|sitemap)>(.*?)</(?:url|sitemap)>", r.text, re.S):
            loc = re.search(r"<loc>\s*([^<\s]+)\s*</loc>", block)
            if not loc:
                continue
            u = htmllib_unescape(loc.group(1))
            mod = re.search(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", block)
            if "<sitemap>" in r.text and re.search(r"\.xml(\?|$)", u):
                out += self.sitemap_urls(u, depth + 1)
                continue
            if mod:
                try:
                    t = datetime.fromisoformat(mod.group(1).replace("Z", "+00:00"))
                    self.lastmod[u] = t if t.tzinfo else t.replace(tzinfo=JST)
                except ValueError:
                    pass
            out.append(u)
        return out

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def fetch(self, url: str, render: str = "", recheck_days: float = 0) -> Page | None:
        if not self.robots_ok(url):
            self.stats["robots_blocked"] += 1
            return None
        cache_file = self._cache_path(url)
        cached: dict[str, Any] | None = None
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached and self._cache_is_fresh(url, cached, recheck_days):
            self.stats["cache_fresh"] += 1
            return Page(url, cached["title"], cached["text"], cached["links"], cached["fetched_at"],
                        cached.get("breadcrumbs", []))
        if render == "playwright":
            return self._fetch_rendered(url, cache_file)
        headers: dict[str, str] = {}
        if cached:
            if cached.get("etag"):
                headers["If-None-Match"] = cached["etag"]
            if cached.get("last_modified"):
                headers["If-Modified-Since"] = cached["last_modified"]

        host = urlparse(url).netloc
        response: httpx.Response | None = None
        for attempt in range(HTTP_RETRIES + 1):
            self._wait(host)
            try:
                response = self.client.get(url, headers=headers)
                if response.status_code < 500:
                    break
            except httpx.HTTPError as exc:
                logger.debug("取得失敗 %s: %s", url, exc)
            if attempt < HTTP_RETRIES:
                time.sleep(self.src.rate_limit_sec * (attempt + 1))
        if response is None:
            self.stats["failed"] += 1
            return None
        if response.status_code == 304 and cached:
            self.stats["not_modified"] += 1
            return Page(url, cached["title"], cached["text"], cached["links"], cached["fetched_at"],
                        cached.get("breadcrumbs", []))
        if response.status_code != 200:
            self.stats["failed"] += 1
            logger.debug("HTTP %d: %s", response.status_code, url)
            return None
        ctype = response.headers.get("content-type", "")
        body = response.text
        if "html" in ctype:
            title, text = extract_html(body, url)
        elif "xml" in ctype or "atom" in ctype:
            title, text = "", ""  # フィード（新着・行事）はリンクだけ辿る
        else:
            return None
        page = Page(
            url=str(response.url) if self.allowed_domain(str(response.url)) else url,
            title=title,
            text=text,
            links=extract_links(body, str(response.url)),
            fetched_at=datetime.now(JST).isoformat(timespec="seconds"),
            breadcrumbs=extract_breadcrumbs(body) if "html" in ctype else [],
        )
        self.stats["fetched"] += 1
        self._save_cache(cache_file, url, page, response.headers.get("etag"),
                         response.headers.get("last-modified"))
        return page


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#][^"']*)["']""", re.I)
# サイト名の接尾辞（「… - 江田島市役所ホームページ」など）を落とす
_TITLE_SUFFIX_RE = re.compile(r"\s*[-|｜–—]\s*[^-|｜–—]*(ホームページ|公式サイト|公式ホームページ)\s*$")


def extract_html(html: str, url: str) -> tuple[str, str]:
    """本文（Markdown）とタイトルを抜く。ナビ・フッタは trafilatura が落とす。"""
    import trafilatura

    m = _TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    title = _TITLE_SUFFIX_RE.sub("", title)
    text = trafilatura.extract(
        html, url=url, output_format="markdown", include_tables=True, include_links=False,
        include_images=False, include_comments=False, favor_precision=True,
    ) or ""
    return title, clean_markdown(text)


def clean_markdown(text: str) -> str:
    """強調記号を落とし、全角・半角の空白の連続を詰める（見出しの # は節分けに使うので残す）。"""
    text = text.replace("**", "")
    text = re.sub(r"[ \t\u3000]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_TAG_RE = re.compile(r"<[^>]+>")
_BREADCRUMB_RE = re.compile(r"(トップページ\s*(?:&gt;|>|＞|›)[^\n]{0,300})")


def extract_breadcrumbs(html: str) -> list[str]:
    """「トップページ > …」の行（1 ページに複数あることがある）。"""
    import html as htmllib

    text = htmllib.unescape(_TAG_RE.sub("", re.sub(r"</(li|p|div|nav|ol|ul)>", "\n", html)))
    text = re.sub(r"\s*\n\s*(?=[>＞›])", " ", text)  # li ごとに改行された区切りをつなぐ
    return [re.sub(r"\s+", " ", m).strip() for m in _BREADCRUMB_RE.findall(text)][:5]


_XML_LINK_RE = re.compile(r"<link>\s*(https?://[^<\s]+)\s*</link>|rdf:about=\"(https?://[^\"]+)\"")


def htmllib_unescape(text: str) -> str:
    import html as htmllib

    return htmllib.unescape(text)


def extract_links(html: str, base: str) -> list[str]:
    out: list[str] = []
    hrefs = _HREF_RE.findall(html) + [a or b for a, b in _XML_LINK_RE.findall(html)]
    for href in hrefs:
        href = htmllib_unescape(href)
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        url, _ = urldefrag(urljoin(base, href.strip()))
        if url.startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out


def crawl(src: Sources, crawler: Crawler) -> list[Page]:
    """seeds から幅優先で辿る（sitemap の seed は載っている URL を候補にする）。

    各 seed の include に合い、範囲（scope_breadcrumbs）・本文の条件（require_text）を満たし、
    除外に当たらないページを集める。ページには seed の trust / label を付ける。
    """
    pages: dict[str, Page] = {}
    fetched: dict[str, Page | None] = {}  # seed をまたいで同じページを取り直さない
    for seed in src.seeds:
        seen: set[str] = set()
        seed_fetches = 0
        if seed.sitemap:
            urls = [u for u in crawler.sitemap_urls(seed.url)
                    if crawler.allowed_domain(u) and seed.matches(seed.include, u)]
            logger.info("sitemap %s: 候補 %d 件", seed.url, len(urls))
            queue: deque[tuple[str, int]] = deque((u, seed.depth) for u in urls)
        else:
            queue = deque([(seed.url, 0)])
        while queue and len(pages) < src.max_pages and len(fetched) < src.max_fetches:
            if seed.max_fetches and seed_fetches >= seed.max_fetches:
                logger.info("seed %s: 取得の上限 %d に達しました", seed.url, seed.max_fetches)
                break
            url, depth = queue.popleft()
            if url in seen or not crawler.allowed_domain(url):
                continue
            if seed.matches(seed.exclude, url):
                continue
            seen.add(url)
            if url not in fetched:
                fetched[url] = crawler.fetch(url, seed.render, seed.recheck_days)
                seed_fetches += 1
                if len(fetched) % 25 == 0:
                    logger.info("取得 %d 件（採用 %d）: %s", len(fetched), len(pages), crawler.stats)
            page = fetched[url]
            if page is None:
                continue
            is_seed = url == seed.url and not seed.sitemap
            if not is_seed and (_excluded(page, src) or not seed.in_scope(page)):
                continue  # 範囲外・除外のページは採用せず、そこから辿らない
            is_target = not is_seed and (not seed.include or seed.matches(seed.include, url))
            if seed.require_text and not any(k in page.text for k in seed.require_text):
                is_target = False
            if is_target and page.text and url not in pages:
                page.trust = seed.trust or src.default_trust
                page.label = seed.label or src.default_label
                pages[url] = page
            if depth < seed.depth:
                for link in page.links:
                    if (link not in seen and crawler.allowed_domain(link)
                            and not seed.matches(seed.exclude, link)):
                        queue.append((link, depth + 1))
    return list(pages.values())


def pages_from_cache(pack: Path, crawler: Crawler, src: Sources) -> list[Page]:
    """今のパックに入っているページを、取得し直さずにキャッシュから読む。"""
    import pyarrow.parquet as pq

    table = pq.read_table(pack / "chunks.parquet").to_pylist()
    meta = {r["source_url"]: (r.get("trust", ""), r.get("label", "")) for r in table}
    urls = list(meta)
    pages: list[Page] = []
    for url in urls:
        f = crawler._cache_path(url)
        if f.is_file():
            c = json.loads(f.read_text(encoding="utf-8"))
            page = Page(url, c["title"], c["text"], c["links"], c["fetched_at"],
                        c.get("breadcrumbs", []))
            page.trust, page.label = meta[url]
            if not page.label:
                page.trust, page.label = src.default_trust, src.default_label
            if not _excluded(page, src):
                pages.append(page)
    logger.info("キャッシュから %d ページ（パックには %d ページ）", len(pages), len(urls))
    return pages


def _excluded(page: Page, src: Sources) -> bool:
    """個人名を含みうるページ（議会・入札・人事など）を外す。タイトルとパンくずで判定。"""
    if any(k in page.title for k in src.exclude_title_keywords):
        return True
    return any(k in b for b in page.breadcrumbs for k in src.exclude_breadcrumb_keywords)


# ----------------------------------------------------------------------
# チャンク
# ----------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    source_url: str
    title: str
    fetched_at: str
    text: str
    trust: str = ""
    label: str = ""


def split_sections(markdown: str) -> list[str]:
    """見出し（# で始まる行）ごとに節に分ける。見出し行は節の先頭に残す。"""
    sections: list[list[str]] = [[]]
    for line in markdown.splitlines():
        if line.lstrip().startswith("#") and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(s).strip() for s in sections if "\n".join(s).strip()]


def chunk_text(markdown: str) -> list[str]:
    """節を基本に CHUNK_MIN〜CHUNK_MAX 字にまとめる。長い節は重なり付きで切る。"""
    pieces: list[str] = []
    buf = ""
    for section in split_sections(markdown):
        section = re.sub(r"\n{3,}", "\n\n", section)
        if len(buf) + len(section) + 1 <= CHUNK_MAX:
            buf = f"{buf}\n{section}".strip()
            if len(buf) >= CHUNK_MIN:
                pieces.append(buf)
                buf = ""
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if len(section) <= CHUNK_MAX:
            buf = section
            continue
        start = 0
        while start < len(section):
            end = min(len(section), start + CHUNK_MAX)
            if end < len(section):  # 文の切れ目（。や改行）で切る
                cut = max(section.rfind("。", start + CHUNK_MIN, end), section.rfind("\n", start + CHUNK_MIN, end))
                if cut > start:
                    end = cut + 1
            pieces.append(section[start:end].strip())
            if end >= len(section):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    if buf:
        pieces.append(buf)
    return [p for p in pieces if len(p) >= CHUNK_DROP_BELOW]


# ---- 人名の除去（知識パックに人名を入れない）----
# 分類での除外（議会・人事など）をすり抜けた人名を、チャンクにする前に伏せる。
# 機械的な規則なので完全ではない（docs/WP03_result.md）。
_NAME = r"[一-龥々\u3400-\u4dbf\uf900-\ufaff]{1,3}[ 　]+[一-龥々\u3400-\u4dbf\uf900-\ufaffぁ-ゖ]{1,4}"
_ROLE = (r"(?:市長|町長|副市長|副町長|教育長|院長|副院長|看護部長|部長|課長|所長|会長|副会長|理事長|"
         r"代表|代表者|委員長|MC|ＭＣ|司会|講師|出演|出演者|考案者|作者)")
REDACTED = "（氏名省略）"
# 1) 役職のあとに続く氏名（「江田島市長　山田　太郎」「副院長（兼）看護部長 山田 花子」「考案者 山田花子」）
_ROLE_NAME_RE = re.compile(
    rf"({_ROLE}(?:（[^）]{{1,6}}）{_ROLE})?(?:[ 　:：]*(?={_NAME})|[ 　:：]+(?=[一-龥々]{{2,3}}[一-龥々ぁ-ゖ]{{1,3}}(?:$|[\s、。／/（(【]))))"
    rf"(?:{_NAME}|[一-龥々]{{2,3}}[一-龥々ぁ-ゖ]{{1,3}})",
    re.M,
)
# 2) 敬称・役職が後ろに付く氏名（「山田 太郎さん」「山田 太郎 議員」）
_NAME_TITLE_RE = re.compile(rf"(?<![一-龥々]){_NAME}(?=[ 　]*(?:さん|氏|様|先生|議員|委員))")
# 3) 氏名だけを ／・、 で並べた行（出演者・MC の一覧）
_NAME_LIST_RE = re.compile(rf"^[ 　]*{_NAME}(?:[ 　]*[／/・、,，][ 　]*{_NAME})+[ 　]*$", re.M)


# 5) 空白なしで敬称が付く氏名（「小杉怜子さん」）。漢字 3〜5 字だけ（「皆さん」「社協さん」「家族の皆様」は当たらない）
_KANJI = "一-龥々\u3400-\u4dbf\uf900-\ufaff"  # 﨑・髙 などの互換漢字も含める
_NAME_NOSPACE_RE = re.compile(rf"(?<![{_KANJI}])[{_KANJI}]{{3,5}}(?=(?:さん|氏|様)(?![まぽぺ々]))")
# 人名ではない「〇〇さん・様」（利用者様・民生委員さん・事業者さん・肉屋さん など）
_NOT_NAME_SUFFIX = ("者", "員", "屋", "業", "家", "族", "方", "人", "生", "長", "師", "体", "手", "主", "店", "所",
                    "局", "様", "客", "達", "皆", "供")
# 4) 伏せた氏名の直後の読み仮名（「（氏名省略）（やまだ たろう）」）と、並べた名（「（氏名省略）さん・花子さん」）
_AFTER_RE = re.compile(rf"{re.escape('（氏名省略）')}(?:[（(][ぁ-ゖー 　]{{2,16}}[）)]|((?:さん|氏)?)・[一-龥々ぁ-ゖ]{{1,4}}(さん|氏))")


def redact_names(text: str) -> str:
    text = _NAME_LIST_RE.sub(REDACTED, text)
    text = _ROLE_NAME_RE.sub(lambda m: m.group(1) + REDACTED, text)
    text = _NAME_TITLE_RE.sub(REDACTED, text)
    text = _NAME_NOSPACE_RE.sub(
        lambda m: m.group(0) if m.group(0).endswith(_NOT_NAME_SUFFIX) else REDACTED, text)
    return _AFTER_RE.sub(lambda m: REDACTED + (m.group(1) or ""), text)


def clean_title(title: str) -> str:
    """タイトルの後ろに付くサイト名・分類（「… | 買う | 江田島市観光協会【公式】…」）を落とす。"""
    t = htmllib_unescape(title).replace("\xa0", " ")
    t = re.split(r"\s*[|｜]\s*", t, maxsplit=1)[0]
    return re.sub(r"\s+", " ", t).strip() or title


def make_chunks(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_text: set[str] = set()
    for page in pages:
        page.title = clean_title(page.title)
        for i, text in enumerate(chunk_text(redact_names(page.text))):
            key = hashlib.sha1(text.encode()).hexdigest()
            if key in seen_text:  # 複数ページに出る同じ文面（共通の案内など）は 1 つにする
                continue
            seen_text.add(key)
            cid = hashlib.sha1(f"{page.url}#{i}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, page.url, page.title, page.fetched_at, text,
                                page.trust, page.label))
    return chunks


# ----------------------------------------------------------------------
# 天気（気象庁）
# ----------------------------------------------------------------------

JMA_FORECAST_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/{code}.json"
AMEDAS_LATEST_URL = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
AMEDAS_MAP_URL = "https://www.jma.go.jp/bosai/amedas/data/map/{stamp}.json"


def build_weather(src: Sources, client: httpx.Client) -> dict[str, Any]:
    """気象庁の府県天気予報（天気・降水確率）と、アメダス最寄り観測点の気温の実況。

    予報の気温は代表点（南部=広島、北部=庄原）のものしか無く、拠点から離れているので使わない。
    気温はアメダスの最寄り観測点の最新の実況値にする（sources の weather.amedas）。
    天気の文言は気象庁のまま。
    """
    w = src.weather
    url = JMA_FORECAST_URL.format(code=w["area_code"])
    r = client.get(url)
    r.raise_for_status()
    data = r.json()[0]
    report = datetime.fromisoformat(data["reportDatetime"])
    series = data["timeSeries"]
    sub = str(w["area_sub_code"])
    area = next(a for a in series[0]["areas"] if a["area"]["code"] == sub)
    dates = [datetime.fromisoformat(t).date() for t in series[0]["timeDefines"]]
    pops_series = series[1] if len(series) > 1 else None
    pops_area = (
        next((a for a in pops_series["areas"] if a["area"]["code"] == sub), None)
        if pops_series else None
    )

    region = f"{w['pref_name']}{area['area']['name']}"
    forecasts: list[dict[str, Any]] = []
    for d, weather in zip(dates, area["weathers"], strict=False):
        if d > report.date() + timedelta(days=1):
            continue
        pops: list[dict[str, Any]] = []
        if pops_series and pops_area:
            for t, pop in zip(pops_series["timeDefines"], pops_area.get("pops", []), strict=False):
                start = datetime.fromisoformat(t)
                if start.date() == d and pop not in ("", None):
                    pops.append({"from": start.hour, "to": start.hour + 6, "pop": int(pop)})
        forecasts.append({"date": d.isoformat(), "region": region, "weather": weather, "pops": pops})

    observation = build_observation(w.get("amedas") or {}, client)
    return {
        "source": f"気象庁（{data['publishingOffice']}）",
        "url": url,
        "report_datetime": data["reportDatetime"],
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "forecasts": forecasts,
        "observation": observation,
    }


def build_observation(cfg: dict[str, Any], client: httpx.Client) -> dict[str, Any] | None:
    """アメダスの最新の気温（観測点はアメダス地点一覧 amedastable.json で ID を確かめて sources に書く）。"""
    if not cfg.get("station_id"):
        return None
    try:
        latest = client.get(AMEDAS_LATEST_URL)
        latest.raise_for_status()
        observed = datetime.fromisoformat(latest.text.strip())
        m = client.get(AMEDAS_MAP_URL.format(stamp=observed.strftime("%Y%m%d%H%M%S")))
        m.raise_for_status()
        temp = (m.json().get(str(cfg["station_id"])) or {}).get("temp")
    except (httpx.HTTPError, ValueError):
        logger.warning("アメダスの実況を取得できません（気温なしで続けます）", exc_info=True)
        return None
    # temp = [値, 品質フラグ]。フラグ 0 が正常値
    if not temp or temp[0] is None or temp[1] != 0:
        logger.warning("アメダス %s の気温が欠測です", cfg["station_id"])
        return None
    return {
        "station_id": str(cfg["station_id"]), "station": cfg["station_name"],
        "place": cfg["place_name"], "in_site": bool(cfg.get("in_site", False)),
        "temp": float(temp[0]), "observed_at": observed.isoformat(timespec="minutes"),
    }


# ----------------------------------------------------------------------
# 保存
# ----------------------------------------------------------------------


def embed_and_save(chunks: list[Chunk], out: Path) -> None:
    import lancedb
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    model = SentenceTransformer(EMBED_MODEL, device="cpu")  # GPU の予備枠は使わない
    texts = [f"{DOC_PREFIX}{c.title}\n{c.text}" for c in chunks]
    t0 = time.monotonic()
    vectors = model.encode(
        texts, batch_size=EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False,
    )
    logger.info("埋め込み %d 件: %.1f 秒（CPU）", len(chunks), time.monotonic() - t0)

    table = pa.table({
        "id": [c.id for c in chunks],
        "source_url": [c.source_url for c in chunks],
        "title": [c.title for c in chunks],
        "fetched_at": [c.fetched_at for c in chunks],
        "text": [c.text for c in chunks],
        "trust": [c.trust for c in chunks],
        "label": [c.label for c in chunks],
    })
    pq.write_table(table, out / "chunks.parquet")
    db = lancedb.connect(str(out / "index.lancedb"))
    db.create_table(
        LANCE_TABLE,
        data=pa.table({"id": table["id"], "vector": pa.array(
            [v.tolist() for v in vectors], type=pa.list_(pa.float32(), int(vectors.shape[1]))
        )}),
        mode="overwrite",
    )


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---- 版の管理（無停止差し替え）----
# $DATA/kb/<site> はシンボリックリンクで、実体は $DATA/kb/<site>.versions/<日時>/。
# 新しい版を別ディレクトリに作り終えてからリンクを付け替える（os.replace でアトミック）。
# フロントは manifest.json の更新を見て、再起動なしで読み直す。直前の版は残す（読み込み中の検索のため）。
KEEP_VERSIONS = 2


def versions_dir(root: Path, site: str) -> Path:
    return root / f"{site}.versions"


def current_pack(root: Path, site: str) -> Path:
    """今の版のディレクトリ。古い配置（実体のディレクトリ）なら版の配置に移す。"""
    link = root / site
    vdir = versions_dir(root, site)
    vdir.mkdir(parents=True, exist_ok=True)
    if link.is_dir() and not link.is_symlink():
        legacy = vdir / f"legacy-{datetime.now(JST):%Y%m%d%H%M%S}"
        link.rename(legacy)
        swap_link(link, legacy)
        logger.info("古い配置を版の配置に移しました: %s", legacy)
    if not link.exists():
        first = vdir / f"{datetime.now(JST):%Y%m%d%H%M%S}"
        first.mkdir()
        swap_link(link, first)
    return link.resolve()


def swap_link(link: Path, target: Path) -> None:
    """link を target に付け替える（一時リンクを作って rename。途中で読まれても壊れない）。"""
    tmp = link.with_name(f".{link.name}.swap")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(target, target_is_directory=True)
    os.replace(tmp, link)


def prune_versions(root: Path, site: str) -> None:
    """新しい順に KEEP_VERSIONS 個だけ残す（今の版は必ず残す）。"""
    current = (root / site).resolve()
    versions = sorted(
        (p for p in versions_dir(root, site).iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name.removeprefix("legacy-"), reverse=True,
    )
    for old in versions[KEEP_VERSIONS:]:
        if old.resolve() != current:
            shutil.rmtree(old)


def link_pack(site: str, target: Path) -> None:
    """kb/pack/<site> → 実体へのシンボリックリンク。"""
    link = KB_DIR / "pack" / site
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        link.unlink()
    link.symlink_to(target, target_is_directory=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="井戸端AI 知識パックのビルド")
    ap.add_argument("--site", required=True)
    ap.add_argument("--weather-only", action="store_true", help="天気（weather.json）だけ更新する")
    ap.add_argument("--max-pages", type=int, default=None, help="試運転用に取得ページ数を絞る")
    ap.add_argument("--from-cache", action="store_true",
                    help="取得し直さず、今のパックのページをキャッシュから作り直す（チャンク・人名除去の変更用）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)

    src = load_sources(args.site)
    if args.max_pages:
        src.max_pages = args.max_pages
    root = data_dir() / "kb"
    root.mkdir(parents=True, exist_ok=True)
    pack = current_pack(root, args.site)  # 今の版（天気はここを書き換える）
    link_pack(args.site, root / args.site)

    with httpx.Client(headers={"User-Agent": src.user_agent}, timeout=HTTP_TIMEOUT_SEC) as client:
        weather = build_weather(src, client)
    write_json(pack / "weather.json", weather)
    for f in weather["forecasts"]:
        logger.info("天気: %s %s %s %s", f["date"], f["region"], f["weather"], f["pops"])
    logger.info("気温: %s", weather.get("observation"))
    if args.weather_only:
        return 0

    crawler = Crawler(src, root / "cache" / args.site)
    t0 = time.monotonic()
    try:
        pages = pages_from_cache(pack, crawler, src) if args.from_cache else crawl(src, crawler)
    finally:
        crawler.close()
    logger.info("取得完了: 採用 %d ページ, %s, %.0f 秒", len(pages), crawler.stats, time.monotonic() - t0)
    chunks = make_chunks(pages)
    logger.info("チャンク %d 件", len(chunks))
    if not chunks:
        logger.error("チャンクが 0 件です。sources を見直してください")
        return 1

    # 新しい版を別ディレクトリに作り、できてからリンクを付け替える（フロントは再起動不要）
    version = versions_dir(root, args.site) / f"{datetime.now(JST):%Y%m%d%H%M%S}"
    staging = version.with_name(f".{version.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    embed_and_save(chunks, staging)
    shutil.copy2(pack / "weather.json", staging / "weather.json")
    write_json(staging / "manifest.json", {
        "site": args.site,
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "pages": len(pages),
        "chunks": len(chunks),
        "embedding": {"model": EMBED_MODEL, "query_prefix": QUERY_PREFIX,
                      "doc_prefix": DOC_PREFIX, "normalized": True},
        "crawl_stats": crawler.stats,
        "sources": [{"url": s.url, "depth": s.depth, "include": s.include, "sitemap": s.sitemap,
                     "render": s.render, "trust": s.trust or src.default_trust,
                     "label": s.label or src.default_label} for s in src.seeds],
        "allow_domains": src.allow_domains,
        "chunk_params": {"min": CHUNK_MIN, "max": CHUNK_MAX, "overlap": CHUNK_OVERLAP},
    })
    staging.rename(version)
    swap_link(root / args.site, version)
    prune_versions(root, args.site)
    logger.info("知識パック: %s → %s（%d ページ, %d チャンク）", root / args.site, version.name,
                len(pages), len(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
