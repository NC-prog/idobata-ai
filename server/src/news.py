"""お知らせ（広報、WP11）の記事・選ぶ順番・要約の安全装置。

記事は夜のパック更新（``kb/build.py`` → ``kb/news_classify.py`` → ``kb/news_summarize.py``）が
``$DATA/kb/<site>/news/<id>.json`` に 1 記事 1 ファイルで書く（実体は ``$DATA/kb/<site>.news/``）。
職員パネルの「お知らせ」タブは同じファイルを直接書き換える（git 管理外。同時編集は考えない）。
フロントは起動時と ``NewsConfig.reload_sec`` ごとに読み直す（手続きナレッジと同じ要領）。

状態（status）:
    auto（自動で出す）/ stopped（職員が止めた）/ priority（職員が優先にした）/ held（要約が安全装置を通らず保留）
held は要約の検査の結果で決まり、要約が通れば auto に戻る。stopped と priority は職員の選択で、検査では変えない。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import yaml

from .config import NewsConfig
from .knowledge import JST, Reference

logger = logging.getLogger(__name__)

STATUSES = ("auto", "stopped", "priority", "held")
STATUS_LABELS = {"auto": "自動", "stopped": "止めた", "priority": "優先", "held": "保留"}
CATEGORIES = ("通行止め", "ライフライン", "施設", "イベント", "募集", "健康", "防災", "ごみ", "その他")
OTHER = "その他"
NEWS_TRUST = "news_article"

_ZEN = str.maketrans("０１２３４５６７８９：", "0123456789:")
# 本文の「10:00」「10：30」を「10時」「10時30分」に読み替えた写しも根拠に使う（要約は話し言葉で書くため）
_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})")
# 読む日でずれる言い方（要約は夜に作り、何日か後に読む）
_RELATIVE_DAY_RE = re.compile(r"今日|本日|明日|あした|昨日|きのう|今週|来週|今月|来月|今朝|今夜|今晩|今度の")
# 日付の直後の「から」「より」「以降」（続いている期間の始まり）
_OPEN_START_RE = re.compile(r"\s*(?:[（(][^）)]{1,4}[）)])?\s*(?:から|より|以降)")
# 年の無い日付（要約は「11月21日」と書く）
_MD_RE = re.compile(r"(\d{1,2})月(\d{1,2})日")


def news_id(url: str) -> str:
    """記事の id（URL のハッシュ）。"""
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def news_dir(data_dir: Path, site: str, cfg: NewsConfig) -> Path:
    """記事の置き場（$DATA/kb/<site>/news。パックの版の中のリンクで、実体は $DATA/kb/<site>.news）。"""
    return data_dir / "kb" / site / cfg.subdir


@dataclass
class NewsRecord:
    """お知らせ 1 件（A-1 のレコード）。"""

    id: str
    title: str
    url: str
    published: str  # 掲載日（YYYY-MM-DD）。一覧・ページに無ければ取得日
    body: str = ""
    image: str | None = None  # 記事の置き場からの相対パス（<id>.jpg）
    image_src: str | None = None  # 元の URL
    category: str = OTHER
    relevance: str = ""  # introduce | skip（分類の前は空）
    relevance_reason: str = ""
    relevance_by: str = ""  # rule | llm
    summary: str = ""
    summary_checks: dict[str, Any] = field(default_factory=dict)  # {"result": ok|held, "reasons": [...], "speech_sec": 秒}
    period: dict[str, str] = field(default_factory=dict)  # {"start": YYYY-MM-DD, "end": YYYY-MM-DD}（取れたときだけ）
    expires_at: str = ""  # この日から候補にしない（YYYY-MM-DD）
    status: str = "auto"
    image_hidden: bool = False
    # 記事のページそのものから画像を探したか（サイトが止まっていてキャッシュの本文で作ったときは False）
    image_checked: bool = False
    fetched_at: str = ""  # 初めて取った日時
    updated_at: str = ""  # 最後に本文を取り直した日時
    content_hash: str = ""
    shown_count: int = 0
    edited_by_staff: bool = False
    edited_by: str = ""
    # 職員が「確認済みとして保存」した（WP13）: ログイン名・日時・そのときの本文のハッシュ。
    # 日付・期限・金額・支払方法の本文照合を外して保存し、夜の再判定で上書きしない（手続きナレッジの「承認」と同じ考え方）
    verified_by: str = ""
    verified_at: str = ""
    verified_hash: str = ""
    label: str = ""  # 出どころの言い方（町のホームページ など）
    # 分類・要約をした本文のハッシュ（content_hash と違えば夜に作り直す）
    classified_hash: str = ""
    summarized_hash: str = ""

    @property
    def held(self) -> bool:
        return self.summary_checks.get("result") == "held"

    @property
    def verified(self) -> bool:
        """職員が確認済みとして保存した要約か。"""
        return bool(self.verified_by)

    @property
    def verified_stale(self) -> bool:
        """確認済みの後に記事の本文が変わった（職員に確かめ直してもらう印。要約は使い続ける）。"""
        return self.verified and self.verified_hash != self.content_hash

    def expired(self, today: date) -> bool:
        return bool(self.expires_at) and date.fromisoformat(self.expires_at) <= today

    def published_date(self) -> date:
        try:
            return date.fromisoformat(self.published[:10])
        except ValueError:
            return date.min


def record_from_dict(d: dict[str, Any]) -> NewsRecord:
    known = {f for f in NewsRecord.__dataclass_fields__}
    return NewsRecord(**{k: v for k, v in d.items() if k in known})


def record_to_dict(r: NewsRecord) -> dict[str, Any]:
    return asdict(r)


def save_record(directory: Path, r: NewsRecord) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{r.id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record_to_dict(r), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_records(directory: Path) -> list[NewsRecord]:
    out: list[NewsRecord] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("."):
            continue  # 収集の結果（.fetch_status.json）は記事ではない（pathlib の glob は「.」で始まる名前にも当たる）
        try:
            out.append(record_from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            logger.warning("お知らせの記事を読めません: %s", path.name, exc_info=True)
    return out


def refresh_status(r: NewsRecord) -> None:
    """要約の検査の結果を状態に映す（職員の止めた・優先は変えない）。"""
    if r.status in ("auto", "held"):
        r.status = "held" if r.held else "auto"


@dataclass(frozen=True)
class NewsRules:
    """分類の語と重み（server/context/news.yaml）。"""

    skip: tuple[str, ...]
    introduce: tuple[str, ...]
    categories: tuple[tuple[str, tuple[str, ...]], ...]
    weights: dict[str, int]
    period_words: tuple[str, ...]
    deadline_words: tuple[str, ...] = ()

    def category(self, title: str, body_head: str) -> str:
        text = f"{title}\n{body_head}"
        for name, words in self.categories:
            if any(w in title for w in words):
                return name
        for name, words in self.categories:
            if any(w in text for w in words):
                return name
        return OTHER

    def weight(self, category: str) -> int:
        return int(self.weights.get(category, self.weights.get(OTHER, 1)))


def load_rules(path: Path) -> NewsRules:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cats = tuple((str(c["name"]), tuple(str(w) for w in c.get("words") or []))
                 for c in data.get("categories") or [] if isinstance(c, dict) and c.get("name"))
    for name, _ in cats:
        if name not in CATEGORIES:
            raise ValueError(f"news.yaml の分類は {CATEGORIES} のどれかです: {name}")
    return NewsRules(
        skip=tuple(str(w) for w in data.get("skip") or []),
        introduce=tuple(str(w) for w in data.get("introduce") or []),
        categories=cats,
        weights={str(k): int(v) for k, v in (data.get("weights") or {}).items()},
        period_words=tuple(str(w) for w in data.get("period_words") or []),
        deadline_words=tuple(str(w) for w in data.get("deadline_words") or []),
    )


def is_candidate(r: NewsRecord, today: date) -> bool:
    """紹介してよい記事か（紹介する分類・要約が通った・止めていない・期限内）。"""
    return (r.relevance == "introduce" and bool(r.summary) and not r.held
            and r.status in ("auto", "priority") and not r.expired(today))


def order_candidates(records: Iterable[NewsRecord], rules: NewsRules, today: date) -> list[NewsRecord]:
    """紹介の順番（B-1）。優先（掲載日の新しい順）→ 分類の重み → 掲載日の新しい順 → 紹介した回数の少ない順。"""
    cands = [r for r in records if is_candidate(r, today)]
    return sorted(cands, key=lambda r: (
        0 if r.status == "priority" else 1,
        -rules.weight(r.category) if r.status != "priority" else 0,
        -r.published_date().toordinal(),
        r.shown_count,
        r.id,
    ))


def spoken_date(published: date, today: date, cfg: NewsConfig) -> str:
    """掲載日の言い方（今日・昨日・N月N日）。"""
    if published == today:
        return cfg.today_word
    if published == today - timedelta(days=1):
        return cfg.yesterday_word
    return f"{published.month}月{published.day}日"


def spoken_body(body: str) -> str:
    """根拠の照合に使う本文の写し（全角数字を半角に、「10:00」を「10時」に）。"""
    text = body.translate(_ZEN)
    return _CLOCK_RE.sub(lambda m: f"{int(m[1])}時" + (f"{int(m[2])}分" if m[2] != "00" else ""), text)


class TextChecker(Protocol):
    """読み上げ前の安全装置（Dialogue.fixed_text_problems）。"""

    def fixed_text_problems(self, text: str, refs: list[Reference], stale_future: bool = True) -> list[str]:
        ...


def summary_problems(summary: str, record: NewsRecord, checker: TextChecker, today: date,
                     claim_re: re.Pattern[str], future_re: re.Pattern[str]) -> list[str]:
    """要約の安全装置（A-3）。保存の前に通す（夜の要約・職員パネルの編集の両方）。

    応答と同じ安全装置（dialogue.py）に加えて、要約はどの文でも単位つきの数・支払方法の語が本文にそのまま有ること、
    年の無い日付（「11月21日」）も掲載日から年を決めて見る:
    - 過ぎた日付を、これからのこと（future_re。「始まります」「開催します」）として言う文
    - 過ぎた日付だけの文（「10月4日は当番です」）。ただし「10月1日から開始しています」のように、過ぎた日付が
      続いている期間の始まり（〜から・より・以降）なら言ってよい
    - 今日・明日などの言い方（要約は夜に作って何日か後に読むので、ずれる）
    """
    text = summary.strip()
    if not text:
        return ["要約が空"]
    body = spoken_body(f"{record.title}\n{record.body}")
    refs = [Reference(record.title, record.url, body, 1.0, 0.0, NEWS_TRUST, record.label)]
    problems = checker.fixed_text_problems(text, refs, stale_future=False)
    flat = re.sub(r"\s", "", body)
    missing = [c for c in claim_re.findall(spoken_body(text)) if c not in flat]
    if missing:
        problems.append("本文に無い数・日付・支払方法")
    if _RELATIVE_DAY_RE.search(text):
        problems.append("今日・明日などの言い方（読む日でずれる）")
    published = record.published_date()
    for sentence in re.split(r"(?<=[。！？!?])", text):
        plain = sentence.translate(_ZEN)
        days = [(resolve_month_day(int(m[1]), int(m[2]), published), m.end()) for m in _MD_RE.finditer(plain)]
        past = [(d, end) for d, end in days if d is not None and d < today]
        if not past or len(past) < len([d for d, _ in days if d is not None]):
            continue  # 過ぎた日付が無い、またはこれから先の日付もある（期間）
        if future_re.search(sentence):
            problems.append("過ぎた日付をこれからと言う")
        elif not any(_OPEN_START_RE.match(plain, end) for _, end in past):
            problems.append("過ぎた日付だけの文")
    return list(dict.fromkeys(problems))


def resolve_month_day(month: int, day: int, published: date) -> date | None:
    """年の無い「M月D日」を掲載日から決める（掲載日より半年以上前の月なら翌年）。"""
    if published == date.min:
        return None
    year = published.year + (1 if month < published.month - 5 else 0)
    try:
        return date(year, month, day)
    except ValueError:
        return None


_ZERO_MIN_RE = re.compile(r"([0-9０-９]{1,2})時[0０]{2}分")
FETCH_STATUS_FILE = ".fetch_status.json"


def tidy_summary(text: str) -> str:
    """要約の後処理（読み上げ・字幕とも）: 「10時00分」を「10時」に。"""
    return _ZERO_MIN_RE.sub(r"\1時", text)


def read_fetch_status(directory: Path) -> dict[str, Any] | None:
    """夜の収集の最後の結果（kb/news_collect.py が書く）。無ければ None。"""
    path = directory / FETCH_STATUS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def qr_data_url(url: str) -> str:
    """記事の URL の QR コード（PNG の data URL。誤り訂正 M。segno、BSD-3）。"""
    import segno

    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=8, border=2)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class NewsBook:
    """拠点の記事（フロントが持つ。reload_sec ごとにディレクトリの変化を見て読み直す）。"""

    def __init__(self, directory: Path, cfg: NewsConfig, rules: NewsRules,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = lambda: datetime.now(JST)) -> None:
        self.directory = directory
        self.cfg = cfg
        self.rules = rules
        self._clock = clock
        self._now = now
        self._records: dict[str, NewsRecord] = {}
        self._stamp: tuple[int, float] | None = None
        self._checked = -1e18
        self._logged_counts: tuple[int, int] | None = None
        self.reload(force=True)

    def _dir_stamp(self) -> tuple[int, float]:
        if not self.directory.is_dir():
            return (0, 0.0)
        files = [f for f in self.directory.glob("*.json") if not f.name.startswith(".")]
        return (len(files), max((f.stat().st_mtime for f in files), default=0.0))

    def reload(self, force: bool = False) -> bool:
        """変わっていれば読み直す（force で必ず）。読み直したら True。"""
        now = self._clock()
        if not force and now - self._checked < self.cfg.reload_sec:
            return False
        self._checked = now
        stamp = self._dir_stamp()
        if not force and stamp == self._stamp:
            return False
        self._stamp = stamp
        self._records = {r.id: r for r in load_records(self.directory)}
        counts = (len(self._records), len(self.candidates()))
        # 件数が変わったときだけ INFO（読み直しは 10 秒ごとに起きうるので、同じ件数は DEBUG。実機 2026-10-24）
        level = logging.INFO if counts != self._logged_counts else logging.DEBUG
        logger.log(level, "お知らせ: %d 件（紹介の候補 %d 件）", *counts)
        self._logged_counts = counts
        return True

    def records(self) -> list[NewsRecord]:
        return list(self._records.values())

    def get(self, rid: str) -> NewsRecord | None:
        return self._records.get(rid)

    def candidates(self) -> list[NewsRecord]:
        return order_candidates(self._records.values(), self.rules, self._now().date())

    def next_candidate(self) -> NewsRecord | None:
        self.reload()
        cands = self.candidates()
        return cands[0] if cands else None

    def save(self, r: NewsRecord) -> None:
        save_record(self.directory, r)
        self._records[r.id] = r
        self._stamp = self._dir_stamp()

    def mark_shown(self, rid: str) -> None:
        r = self._records.get(rid)
        if r is None:
            return
        r.shown_count += 1
        self.save(r)

    def image_path(self, rid: str) -> Path | None:
        """配信してよい画像のファイル（職員が画像を止めた記事・画像の無い記事は None）。"""
        r = self._records.get(rid)
        if r is None or r.image_hidden or not r.image:
            return None
        path = self.directory / r.image
        return path if path.is_file() and path.parent == self.directory else None

    def as_references(self, query: str) -> list[Reference]:
        """紹介する記事を、行事の問いの資料にする（WP13 B。名詞の足切りは呼び出し側）。"""
        del query
        self.reload()
        return [Reference(r.title, r.url, tidy_summary(r.summary) or r.body[:300], 1.0, 0.0, "news", r.label)
                for r in self.candidates()]

    def layout_message(self, r: NewsRecord) -> dict[str, Any]:
        """エッジへ送る紹介の構図（C-3 の 2）。"""
        image = f"/edge/news/{r.id}.jpg" if self.image_path(r.id) is not None else None
        return {"type": "layout", "mode": "news", "id": r.id, "title": r.title, "image": image,
                "qr": qr_data_url(r.url), "url": r.url, "category": r.category, "label": r.label}
