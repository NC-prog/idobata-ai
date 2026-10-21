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
        self._last_request: dict[str, float] = {}
        self.stats = {"fetched": 0, "not_modified": 0, "failed": 0, "robots_blocked": 0}

    def close(self) -> None:
        self.client.close()

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

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def fetch(self, url: str) -> Page | None:
        if not self.robots_ok(url):
            self.stats["robots_blocked"] += 1
            return None
        cache_file = self._cache_path(url)
        cached: dict[str, Any] | None = None
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
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
        cache_file.write_text(json.dumps({
            "url": url, "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "title": page.title, "text": page.text, "links": page.links,
            "breadcrumbs": page.breadcrumbs,
            "fetched_at": page.fetched_at,
        }, ensure_ascii=False), encoding="utf-8")
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


def extract_links(html: str, base: str) -> list[str]:
    out: list[str] = []
    for href in _HREF_RE.findall(html):
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        url, _ = urldefrag(urljoin(base, href.strip()))
        if url.startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out


def crawl(src: Sources, crawler: Crawler) -> list[Page]:
    """seeds から幅優先で辿る。各 seed の include に合うページだけを集める。"""
    pages: dict[str, Page] = {}
    fetched: dict[str, Page | None] = {}  # seed をまたいで同じページを取り直さない
    for seed in src.seeds:
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(seed.url, 0)])
        while queue and len(pages) < src.max_pages and len(fetched) < src.max_fetches:
            url, depth = queue.popleft()
            if url in seen or not crawler.allowed_domain(url):
                continue
            if seed.matches(seed.exclude, url):
                continue
            seen.add(url)
            if url not in fetched:
                fetched[url] = crawler.fetch(url)
                if len(fetched) % 25 == 0:
                    logger.info("取得 %d 件（採用 %d）: %s", len(fetched), len(pages), crawler.stats)
            page = fetched[url]
            if page is None:
                continue
            is_seed = url == seed.url
            if not is_seed and (_excluded(page, src) or not seed.in_scope(page)):
                continue  # 範囲外・除外のページは採用せず、そこから辿らない
            is_target = not is_seed and (not seed.include or seed.matches(seed.include, url))
            if is_target and page.text:
                pages.setdefault(url, page)
            if depth < seed.depth:
                for link in page.links:
                    if (link not in seen and crawler.allowed_domain(link)
                            and not seed.matches(seed.exclude, link)):
                        queue.append((link, depth + 1))
    return list(pages.values())


def pages_from_cache(pack: Path, crawler: Crawler, src: Sources) -> list[Page]:
    """今のパックに入っているページを、取得し直さずにキャッシュから読む。"""
    import pyarrow.parquet as pq

    urls = list(dict.fromkeys(pq.read_table(pack / "chunks.parquet")["source_url"].to_pylist()))
    pages: list[Page] = []
    for url in urls:
        f = crawler._cache_path(url)
        if f.is_file():
            c = json.loads(f.read_text(encoding="utf-8"))
            page = Page(url, c["title"], c["text"], c["links"], c["fetched_at"],
                        c.get("breadcrumbs", []))
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
_NAME = r"[一-龥々]{1,3}[ 　]+[一-龥々ぁ-ゖ]{1,4}"
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


# 4) 伏せた氏名の直後の読み仮名（「（氏名省略）（やまだ たろう）」）と、並べた名（「（氏名省略）さん・花子さん」）
_AFTER_RE = re.compile(rf"{re.escape('（氏名省略）')}(?:[（(][ぁ-ゖー 　]{{2,16}}[）)]|((?:さん|氏)?)・[一-龥々ぁ-ゖ]{{1,4}}(さん|氏))")


def redact_names(text: str) -> str:
    text = _NAME_LIST_RE.sub(REDACTED, text)
    text = _ROLE_NAME_RE.sub(lambda m: m.group(1) + REDACTED, text)
    text = _NAME_TITLE_RE.sub(REDACTED, text)
    return _AFTER_RE.sub(lambda m: REDACTED + (m.group(1) or ""), text)


def make_chunks(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_text: set[str] = set()
    for page in pages:
        for i, text in enumerate(chunk_text(redact_names(page.text))):
            key = hashlib.sha1(text.encode()).hexdigest()
            if key in seen_text:  # 複数ページに出る同じ文面（共通の案内など）は 1 つにする
                continue
            seen_text.add(key)
            cid = hashlib.sha1(f"{page.url}#{i}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, page.url, page.title, page.fetched_at, text))
    return chunks


# ----------------------------------------------------------------------
# 天気（気象庁）
# ----------------------------------------------------------------------

JMA_FORECAST_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/{code}.json"
WEEKDAYS = "月火水木金土日"


def build_weather(src: Sources, client: httpx.Client) -> dict[str, Any]:
    """気象庁の府県天気予報から、今日・明日の 1〜2 文を作る。天気の文言は気象庁のまま。"""
    w = src.weather
    url = JMA_FORECAST_URL.format(code=w["area_code"])
    r = client.get(url)
    r.raise_for_status()
    data = r.json()[0]
    report = datetime.fromisoformat(data["reportDatetime"])
    series = data["timeSeries"]
    area = next(a for a in series[0]["areas"] if a["area"]["code"] == str(w["area_sub_code"]))
    dates = [datetime.fromisoformat(t).date() for t in series[0]["timeDefines"]]
    temps_series = next(
        (ts for ts in series if any(a["area"]["name"] == w["point_name"] for a in ts["areas"])), None
    )
    maxima: dict[Any, str] = {}
    minima: dict[Any, str] = {}
    if temps_series is not None:
        point = next(a for a in temps_series["areas"] if a["area"]["name"] == w["point_name"])
        for t, v in zip(temps_series["timeDefines"], point.get("temps", []), strict=False):
            when = datetime.fromisoformat(t)
            if v in ("", None):
                continue
            if when.hour == 9:
                maxima[when.date()] = v
            elif when.hour == 0 and (when.date() > report.date() or report.hour < 9):
                minima[when.date()] = v  # 発表後の朝の最低気温だけ（当日 11 時以降の 0 時は最高の重複）

    labels = {report.date(): "今日", report.date() + timedelta(days=1): "明日"}
    region = f"{w['pref_name']}{area['area']['name']}"
    lines: list[dict[str, Any]] = []
    for d, weather in zip(dates, area["weathers"], strict=False):
        if d not in labels:
            continue
        text = f"{labels[d]}（{d.month}月{d.day}日）の{region}は「{weather}」"
        temps: list[str] = []
        if d in maxima:
            temps.append(f"最高気温は{maxima[d]}度")
        if d in minima:
            temps.append(f"最低気温は{minima[d]}度")
        if temps:
            text += f"、{w['point_name']}の" + "、".join(temps)
        text += "の予報です。"
        # server/src/knowledge.py の format_weather が、その日の「今日 / 明日」で文を組み直す
        lines.append({
            "date": d.isoformat(), "region": region, "weather": weather,
            "point": w["point_name"], "max": maxima.get(d, ""), "min": minima.get(d, ""),
            "text": text,
        })
    return {
        "source": f"気象庁（{data['publishingOffice']}）",
        "url": url,
        "report_datetime": data["reportDatetime"],
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "forecasts": lines,
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
    pack = root / args.site
    pack.mkdir(parents=True, exist_ok=True)
    link_pack(args.site, pack)

    with httpx.Client(headers={"User-Agent": src.user_agent}, timeout=HTTP_TIMEOUT_SEC) as client:
        weather = build_weather(src, client)
    write_json(pack / "weather.json", weather)
    for f in weather["forecasts"]:
        logger.info("天気: %s", f["text"])
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

    # 作りかけを読ませないよう、別ディレクトリに作ってから差し替える
    staging = root / f".{args.site}.building"
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
        "sources": [{"url": s.url, "depth": s.depth, "include": s.include} for s in src.seeds],
        "allow_domains": src.allow_domains,
        "chunk_params": {"min": CHUNK_MIN, "max": CHUNK_MAX, "overlap": CHUNK_OVERLAP},
    })
    old = root / f".{args.site}.old"
    if old.exists():
        shutil.rmtree(old)
    pack.rename(old)
    staging.rename(pack)
    shutil.rmtree(old)
    logger.info("知識パック: %s（%d ページ, %d チャンク）", pack, len(pages), len(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
