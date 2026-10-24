"""お知らせの分類（WP11 A-2）。夜だけ動く（kb/build.py のお知らせの区画から呼ぶ。単独でも回せる）。

    uv run --project kb python kb/news_classify.py --site etajima

まず規則（server/context/news.yaml の skip / introduce の語。題名を本文より先に見る）。規則で決まらないものだけ
LLM（7560 の Qwen3-8B、思考オフ、temperature 0）に introduce / skip の二択で答えさせ、理由を 1 文。
本文が変わった記事（content_hash が classified_hash と違う）だけやり直す。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

KB_DIR = Path(__file__).resolve().parent
SERVER_DIR = KB_DIR.parent / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from src.config import NewsConfig, config
from src.knowledge import JST
from src.news import NewsRecord, NewsRules, load_records, load_rules, save_record

logger = logging.getLogger("kb.news")

# 規則で本文を見る範囲（先頭の字数）
RULE_BODY_CHARS = 300
# これより短い本文は紹介しない（資料の PDF の一覧だけの記事）
MIN_BODY_CHARS = 60


class NightLLM:
    """夜の分類・要約の LLM（vLLM の OpenAI 互換 API。タイムアウトと再試行つき）。"""

    def __init__(self, cfg: NewsConfig | None = None) -> None:
        self.cfg = cfg or config.news
        self.url = f"{config.llm.gateway_url}/chat/completions"
        self.client = httpx.Client(timeout=self.cfg.llm_timeout_sec)

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": config.llm.chat_model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last: Exception | None = None
        for attempt in range(self.cfg.llm_retries + 1):
            try:
                r = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {config.llm.api_key}"})
                r.raise_for_status()
                return str(r.json()["choices"][0]["message"]["content"] or "").strip()
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"LLM に繋がりません: {last}")

    def close(self) -> None:
        self.client.close()


def rule_relevance(title: str, body: str, rules: NewsRules) -> tuple[str, str] | None:
    """規則での判定（introduce / skip と理由）。決まらなければ None。"""
    head = body[:RULE_BODY_CHARS]
    for text, where in ((title, "題名"), (head, "本文")):
        hit = next((w for w in rules.skip if w in text), None)
        if hit:
            return "skip", f"{where}に「{hit}」"
        hit = next((w for w in rules.introduce if w in text), None)
        if hit:
            return "introduce", f"{where}に「{hit}」"
    return None


def parse_llm_answer(text: str) -> tuple[str, str] | None:
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not lines:
        return None
    head = lines[0].lower()
    label = "introduce" if "introduce" in head else "skip" if "skip" in head else None
    if label is None:
        return None
    reason = lines[1] if len(lines) > 1 else ""
    return label, reason[:120]


def stale(r: NewsRecord, today: date, cfg: NewsConfig) -> bool:
    """掲載から max_age_days 日より前の記事で、期間の終わりがこれから先でもない（新着ではない）。"""
    if (today - r.published_date()).days <= cfg.max_age_days:
        return False
    return not (r.period.get("end") and date.fromisoformat(r.period["end"]) >= today)


def deadline_passed(r: NewsRecord, rules: NewsRules, today: date) -> date | None:
    """申し込みの締め切りの行の日付がすべて過ぎていれば、その最後の日（過ぎていなければ None）。"""
    from news_collect import body_dates

    dates = [d for line in r.body.splitlines() if any(w in line for w in rules.deadline_words)
             for d in body_dates(line, r.published_date())]
    return max(dates) if dates and max(dates) < today else None


def classify_record(r: NewsRecord, rules: NewsRules, llm: NightLLM | None, cfg: NewsConfig,
                    today: date | None = None) -> bool:
    """1 件を分類する（書き換えたら True）。LLM が要るのに無い・失敗したときは分類しない（次の夜にやり直す）。"""
    today = today or datetime.now(JST).date()
    r.category = rules.category(r.title, r.body[:cfg.classify_body_chars])
    ruled = rule_relevance(r.title, r.body, rules)
    closed = deadline_passed(r, rules, today)
    if len(r.body) < MIN_BODY_CHARS:
        ruled = ("skip", "本文がほとんど無い（資料の一覧だけ）")
    elif r.period.get("end") and date.fromisoformat(r.period["end"]) < today:
        ruled = ("skip", "期間が過ぎている")
    elif closed is not None:
        ruled = ("skip", f"申し込みの締め切り（{closed.month}月{closed.day}日）が過ぎている")
    elif stale(r, today, cfg):
        ruled = ("skip", f"掲載から {(today - r.published_date()).days} 日たっている")
    if ruled is not None:
        r.relevance, r.relevance_reason = ruled
        r.relevance_by = "rule"
    else:
        if llm is None:
            return False
        try:
            answer = llm.chat([
                {"role": "system", "content": cfg.classify_prompt},
                {"role": "user", "content": f"題名: {r.title}\n本文: {r.body[:cfg.classify_body_chars]}"},
            ], cfg.classify_max_tokens)
        except RuntimeError:
            logger.warning("分類の LLM に失敗しました（次の夜にやり直します）: %s", r.id)
            return False
        parsed = parse_llm_answer(answer)
        if parsed is None:
            logger.warning("分類の答えを読めません（skip にします）: %s", r.id)
            parsed = ("skip", "答えを読めなかった")
        r.relevance, r.relevance_reason = parsed
        r.relevance_by = "llm"
    r.classified_hash = r.content_hash
    return True


def classify(store: Path, llm: NightLLM | None, cfg: NewsConfig | None = None, force: bool = False) -> dict[str, int]:
    cfg = cfg or config.news
    rules = load_rules(cfg.rules_path)
    stats = {"introduce": 0, "skip": 0, "rule": 0, "llm": 0, "pending": 0}
    for r in load_records(store):
        if force or r.classified_hash != r.content_hash:
            if classify_record(r, rules, llm, cfg):
                save_record(store, r)
            else:
                stats["pending"] += 1
                continue
        if r.relevance:
            stats[r.relevance] += 1
            stats[r.relevance_by or "rule"] += 1
    logger.info("お知らせの分類: %s", stats)
    return stats


def main() -> int:
    import build

    ap = argparse.ArgumentParser(description="お知らせの分類（WP11）")
    ap.add_argument("--site", required=True)
    ap.add_argument("--force", action="store_true", help="分類済みもやり直す（規則を変えたとき）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from news_collect import store_dir

    llm = NightLLM()
    try:
        classify(store_dir(build.data_dir() / "kb", args.site), llm, force=args.force)
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
