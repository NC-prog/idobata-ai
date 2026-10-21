"""調べる（WP04 A-3）。

来訪者が同意したら、許可ドメイン（``kb/sources/<site>.yaml`` の ``lookup``）だけを見に行く。
外に出すのは LLM が作った検索語だけ（本人の事情・氏名・体調は含めない）。検索語はログに出さない。

``lookup`` の各項目:
- ``type: search``: サイト内検索の URL テンプレート（``{q}`` に検索語）。結果ページのリンクを読む
- ``type: list``: 一覧・フィード（お知らせ・イベント）。リンクの文言が検索語に近いものを読む
- ``render: playwright``: ブラウザで描画してから読む（JS で描画するサイト）

合計 timeout_sec で打ち切る。見つからなければ検索語を ``$DATA/kb/<site>.seed_queue.jsonl`` に積み、
翌朝の知識パックの更新（kb/build.py --daily）が取りに行く。
一般の検索エンジン（web_search）は既定 OFF。ON のときだけ、検索語を DuckDuckGo の HTML 版に投げる。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urldefrag, urljoin, urlparse

import httpx
import yaml

from .config import LookupConfig
from .knowledge import _BM25, JST, Reference, bigrams

logger = logging.getLogger(__name__)

_HREF_RE = re.compile(r"<a\s[^>]*href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_XML_ITEM_RE = re.compile(
    r"<(?:item|entry)\b.*?<title[^>]*>(.*?)</title>.*?"
    r"(?:<link[^>]*href=\"([^\"]+)\"|<link>\s*([^<\s]+)\s*</link>)", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class LookupSource:
    name: str
    type: str  # search / list
    url: str
    render: str = ""
    label: str = ""
    trust: str = ""


@dataclass
class LookupResult:
    found: bool
    references: list[Reference] = field(default_factory=list)
    queued: bool = False
    pages_read: int = 0


def load_lookup_sources(path: Path) -> tuple[list[LookupSource], list[str]]:
    """kb/sources/<site>.yaml の lookup と allow_domains。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    default_label = str(data.get("default_label", ""))
    default_trust = str(data.get("default_trust", ""))
    sources = [
        LookupSource(name=str(x.get("name", x["url"])), type=str(x.get("type", "list")), url=str(x["url"]),
                     render=str(x.get("render", "")), label=str(x.get("label", default_label)),
                     trust=str(x.get("trust", default_trust)))
        for x in data.get("lookup") or []
    ]
    return sources, [str(d) for d in data.get("allow_domains") or []]


def _text(html: str) -> str:
    import html as htmllib

    return re.sub(r"\s+", " ", htmllib.unescape(_TAG_RE.sub("", html))).strip()


def candidate_links(page: str, base: str, allow: list[str]) -> list[tuple[str, str]]:
    """ページ（HTML / RSS / Atom）から (URL, 文言) を取り出す。許可ドメインのものだけ。"""
    out: list[tuple[str, str]] = []
    for title, href_attr, href_text in _XML_ITEM_RE.findall(page):
        out.append((href_attr or href_text, _text(title)))
    for href, label in _HREF_RE.findall(page):
        out.append((href, _text(label)))
    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for href, label in out:
        url, _ = urldefrag(urljoin(base, href.strip()))
        if url in seen or urlparse(url).hostname not in allow or not label:
            continue
        seen.add(url)
        links.append((url, label))
    return links


def score_label(label: str, query: str) -> float:
    """リンクの文言と検索語の文字 2-gram の重なり。"""
    q = set(bigrams(query))
    if not q:
        return 0.0
    return len(q & set(bigrams(label))) / len(q)


def chunk_plain(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """本文を知識パックと同じくらいの大きさに切る（見出しが無い本文向けの簡易版）。"""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        cut = text.rfind("。", start + size // 2, end)
        if end < len(text) and cut > start:
            end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if len(c) >= 40]


class Lookup:
    """許可ドメインを、その場で見に行く。"""

    def __init__(self, cfg: LookupConfig, site: str, data_dir: Path,
                 make_query: Callable[[str], Awaitable[str | None]]) -> None:
        self._cfg = cfg
        self._site = site
        # 版のディレクトリ（kb/<site> はリンクで、作り直すと付け替わる）の外に置く
        self._queue_path = data_dir / "kb" / f"{site}.seed_queue.jsonl"
        self._make_query = make_query
        path = cfg.sources_dir / f"{site}.yaml"
        self.sources, self.allow = load_lookup_sources(path) if path.is_file() else ([], [])

    async def run(self, user_text: str) -> LookupResult:
        query = await self._make_query(user_text)
        if not query:
            return LookupResult(found=False)
        try:
            result = await asyncio.wait_for(self._search(query), self._cfg.timeout_sec)
        except TimeoutError:
            logger.info("調べる: %.0f 秒で打ち切りました", self._cfg.timeout_sec)
            result = LookupResult(found=False)
        if not result.found:
            result.queued = self.enqueue(query)
        logger.info("調べる: 見つかった=%s 読んだページ=%d 夜に回す=%s",
                    result.found, result.pages_read, result.queued)
        return result

    def enqueue(self, query: str) -> bool:
        """見つからなかった検索語を、翌朝の知識パックの更新に回す（検索語だけ。発話は残さない）。"""
        try:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            with self._queue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"query": query, "site": self._site,
                                    "queued_at": datetime.now(JST).isoformat(timespec="seconds")},
                                   ensure_ascii=False) + "\n")
            return True
        except OSError:
            logger.warning("seed_queue に書けません: %s", self._queue_path, exc_info=True)
            return False

    async def prepare(self) -> None:
        """描画が要る取得先があれば、ブラウザを先に起こしておく（初回の起動に 2〜3 秒かかる）。"""
        if any(s.render == "playwright" for s in self.sources):
            try:
                await _ensure_browser()
            except Exception:
                logger.warning("描画用のブラウザを起動できません（描画が要る取得先は使えません）", exc_info=True)

    async def _search(self, query: str) -> LookupResult:
        links: list[tuple[float, str, str, LookupSource]] = []
        deadline = time.monotonic() + self._cfg.timeout_sec
        async with httpx.AsyncClient(timeout=self._cfg.timeout_sec, follow_redirects=True,
                                     headers={"User-Agent": self._cfg.user_agent}) as client:
            # 一覧は持ち時間の内に返ったものだけ使う（遅い取得先ひとつで全体を待たない）
            pages = await _gather_within(
                [self._fetch(client, self._source_url(s, query), s.render) for s in self.sources],
                self._cfg.list_timeout_sec)
            for src, page in zip(self.sources, pages, strict=True):
                if not isinstance(page, str):
                    continue
                for url, label in candidate_links(page, self._source_url(src, query), self.allow):
                    s = score_label(label, query)
                    if s > 0:  # 検索結果は少し優先する（合わないナビのリンクまで拾わないよう、合ったものだけ）
                        links.append((s + (0.5 if src.type == "search" else 0.0), url, label, src))
            if self._cfg.web_search_enabled:
                links += await self._web_search(client, query)
            links.sort(key=lambda x: -x[0])
            chosen = links[: self._cfg.max_pages]
            bodies = await _gather_within([self._fetch(client, u, src.render) for _, u, _, src in chosen],
                                          deadline - time.monotonic() - self._cfg.margin_sec)
        chunks: list[tuple[str, str, LookupSource, str]] = []
        for (_, url, label, src), body in zip(chosen, bodies, strict=True):
            if isinstance(body, str):
                chunks += [(url, label, src, c) for c in chunk_plain(_extract(body, url))]
        if not chunks:
            return LookupResult(found=False, pages_read=len(chosen))
        bm = _BM25([f"{label}\n{c}" for _, label, _, c in chunks], 1.5, 0.75)
        ranked = bm.search(query, len(chunks))
        refs: list[Reference] = []
        total = 0
        for i, score in ranked:
            if score < self._cfg.min_score or total >= self._cfg.max_chars:
                break
            url, label, src, text = chunks[i]
            text = text[: self._cfg.max_chars - total]
            refs.append(Reference(label, url, text, 0.0, score, src.trust, src.label))
            total += len(text)
        return LookupResult(found=bool(refs), references=refs, pages_read=len(chosen))

    @staticmethod
    def _source_url(src: LookupSource, query: str) -> str:
        return src.url.replace("{q}", quote(query)) if src.type == "search" else src.url

    async def _fetch(self, client: httpx.AsyncClient, url: str, render: str) -> str:
        if render == "playwright":
            return await _render(url, self._cfg.user_agent, self._cfg.timeout_sec)
        r = await client.get(url)
        r.raise_for_status()
        return r.text

    async def _web_search(self, client: httpx.AsyncClient,
                          query: str) -> list[tuple[float, str, str, LookupSource]]:
        """一般の検索エンジン（既定 OFF）。上位の結果は許可ドメイン外でも読む。"""
        src = LookupSource("web", "search", self._cfg.web_search_url, label="インターネットの検索結果",
                           trust="web")
        try:
            r = await client.get(self._source_url(src, query))
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        out: list[tuple[float, str, str, LookupSource]] = []
        for href, label in _HREF_RE.findall(r.text):
            m = re.search(r"uddg=([^&]+)", href)  # DuckDuckGo の転送リンクから本当の URL を取り出す
            url = unquote(m.group(1)) if m else href
            if url.startswith("http") and "duckduckgo.com" not in url:
                out.append((1.0, url, _text(label), src))
            if len(out) >= self._cfg.web_search_results:
                break
        return out


def _extract(html: str, url: str) -> str:
    import trafilatura

    if html.lstrip().startswith("<?xml") or "<rss" in html[:500]:
        return _text(html)
    return trafilatura.extract(html, url=url, include_tables=True, favor_precision=True) or ""


_browser_lock = asyncio.Lock()
_browser: Any = None


_playwright: Any = None


async def _gather_within(coros: list[Coroutine[Any, Any, str]], timeout: float) -> list[str | BaseException]:
    """coros を並行に走らせ、timeout 秒までに終わったものの結果を返す（残りは取り消して TimeoutError）。"""
    tasks = [asyncio.ensure_future(c) for c in coros]
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout))
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    out: list[str | BaseException] = []
    for t in tasks:
        if t in done and not t.cancelled():
            exc = t.exception()
            out.append(exc if exc is not None else t.result())
        else:
            out.append(TimeoutError())
    return out


async def _ensure_browser() -> Any:
    global _browser, _playwright
    async with _browser_lock:
        if _browser is None:
            from playwright.async_api import async_playwright

            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch()
    return _browser


async def close_browser() -> None:
    """描画用のブラウザを閉じる（プロセス終了の前に）。"""
    global _browser, _playwright
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None


async def _render(url: str, user_agent: str, timeout_sec: float) -> str:
    """Playwright の Chromium で描画した HTML（1 つのブラウザを使い回す）。"""
    browser = await _ensure_browser()
    page = await browser.new_page(user_agent=user_agent)
    try:
        await page.goto(url, wait_until="networkidle", timeout=int(timeout_sec * 1000))
        html: str = await page.content()
        return html
    finally:
        await page.close()
