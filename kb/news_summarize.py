"""お知らせの要約（WP11 A-3）。夜だけ動く（kb/build.py のお知らせの区画から呼ぶ。単独でも回せる）。

    uv run --project kb python kb/news_summarize.py --site etajima

introduce の記事だけ、LLM に 90〜120 字・2〜3 文の話し言葉（案内役の口調）で書かせる。保存の前に読み上げ前の
安全装置（server/src/dialogue.py の Dialogue.fixed_text_problems と server/src/news.py の summary_problems）に通し、
VOICEVOX で合成した長さ（文ごとに合成して足す。WAV は保存しない）が speech_min_sec〜speech_max_sec に収まらなければ
1 回だけ書き直させる。どれかで引っかかれば summary_checks を held にして紹介しない（職員パネルで直せる）。
職員が直した要約（edited_by_staff）は書き直さず、本文が変わったら安全装置だけ掛け直す。
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
import wave
from datetime import date, datetime
from pathlib import Path

import httpx

KB_DIR = Path(__file__).resolve().parent
SERVER_DIR = KB_DIR.parent / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from news_classify import NightLLM

from src.config import NewsConfig, config
from src.dialogue import Dialogue, speech_text
from src.knowledge import JST
from src.news import (
    NewsRecord,
    load_records,
    refresh_status,
    save_record,
    summary_problems,
    tidy_summary,
)
from src.persona import load_persona
from src.site_context import load_facility

logger = logging.getLogger("kb.news")

LENGTH_REASON_SHORT = "読み上げが短すぎる"
LENGTH_REASON_LONG = "読み上げが長すぎる"
LENGTH_REASON_UNKNOWN = "読み上げの長さを測れない"


class _NoLLM:
    """安全装置だけを使う Dialogue に渡す LLM（呼ばれない）。"""

    def chat(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError("安全装置の検査では LLM を呼びません")

    def chat_stream(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("安全装置の検査では LLM を呼びません")


def make_checker(site: str, now: datetime | None = None) -> Dialogue:
    """拠点の固定事実を持った、安全装置だけを使う Dialogue。"""
    facility = load_facility(site, config.asr.site_context_dir)
    at = now or datetime.now(JST)
    return Dialogue(_NoLLM(), facility=facility, persona=load_persona(config.persona), now=lambda: at,
                    avatar_name=config.actions.avatar_name)


def checks_for(summary: str, r: NewsRecord, checker: Dialogue, today: date) -> list[str]:
    return summary_problems(summary, r, checker, today, re.compile(config.dialogue.grounding_claim_pattern),
                            re.compile(config.news.summary_future_pattern))


class SpeechMeter:
    """VOICEVOX で合成した長さ（秒）。文ごとに合成して足す（読み上げも文単位で合成する）。"""

    def __init__(self) -> None:
        self.cfg = config.voicevox
        self.client = httpx.Client(base_url=self.cfg.base_url, timeout=self.cfg.timeout)

    def _post(self, path: str, **kwargs: object) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.client.post(path, **kwargs)  # type: ignore[arg-type]
                r.raise_for_status()
                return r
            except httpx.HTTPError as exc:
                last = exc
                time.sleep(self.cfg.retry_wait_sec * (attempt + 1))
        raise RuntimeError(f"VOICEVOX に繋がりません: {last}")

    def seconds(self, text: str) -> float:
        total = 0.0
        for sentence in re.split(r"(?<=[。！？!?])", text):
            spoken = speech_text(sentence)
            if not spoken:
                continue
            q = self._post("/audio_query", params={"text": spoken, "speaker": self.cfg.speaker_id}).json()
            q.update(speedScale=self.cfg.speed_scale, pitchScale=self.cfg.pitch_scale,
                     intonationScale=self.cfg.intonation_scale, volumeScale=self.cfg.volume_scale,
                     postPhonemeLength=self.cfg.post_phoneme_length)
            wav = self._post("/synthesis", params={"speaker": self.cfg.speaker_id}, json=q).content
            with wave.open(io.BytesIO(wav)) as w:
                total += w.getnframes() / w.getframerate()
        return round(total, 1)

    def close(self) -> None:
        self.client.close()


def clean_summary(text: str, cfg: NewsConfig) -> str:
    """LLM の要約の後始末（前置きの記号・改行・引用符と、画面に出す「くわしくはホームページで」の類の文を落とす）。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*(要約|お知らせ)\s*[:：]\s*", "", text.strip())
    text = re.sub(r"\s*\n\s*", "", text.strip("「」『』\"' \n"))
    drop = re.compile(cfg.summary_drop_pattern)
    return tidy_summary("".join(x for x in re.split(r"(?<=[。！？!?])", text) if x.strip() and not drop.search(x)))


def target_chars(summary: str, sec: float, cfg: NewsConfig) -> int:
    """読み上げの長さを範囲の真ん中にする字数（1 回目の読み上げの速さから見積もる）。"""
    if sec <= 0:
        return cfg.summary_max_chars
    return round(len(summary) * (cfg.speech_min_sec + cfg.speech_max_sec) / 2 / sec / 5) * 5


def length_problem(sec: float | None, cfg: NewsConfig) -> str | None:
    if sec is None:
        return LENGTH_REASON_UNKNOWN
    if sec < cfg.speech_min_sec:
        return LENGTH_REASON_SHORT
    if sec > cfg.speech_max_sec:
        return LENGTH_REASON_LONG
    return None


def summarize_record(r: NewsRecord, llm: NightLLM, meter: SpeechMeter | None, checker: Dialogue,
                     cfg: NewsConfig, today: date) -> bool:
    """1 件の要約を作って検査する（書き換えたら True）。LLM が失敗したら何もしない（次の夜にやり直す）。"""
    persona = checker._persona
    speaking = "".join(f"- {x}\n" for x in (persona.speaking if persona else ()))
    system = cfg.summarize_prompt.format(min_chars=cfg.summary_min_chars, max_chars=cfg.summary_max_chars,
                                         speaking=speaking)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"【記事】\n題名: {r.title}\n{r.body[:cfg.summarize_body_chars]}\n\n"
                                            f"【日付の一覧】\n{date_list(r, today, cfg)}"}]
    sec: float | None = None
    problems: list[str] = []
    summary = ""
    for attempt in range(2):
        try:
            summary = clean_summary(llm.chat(messages, cfg.summarize_max_tokens), cfg)
        except RuntimeError:
            logger.warning("要約の LLM に失敗しました（次の夜にやり直します）: %s", r.id)
            return False
        problems = checks_for(summary, r, checker, today)
        sec = measure(meter, summary)
        length = length_problem(sec, cfg)
        if length is None or attempt == 1 or length == LENGTH_REASON_UNKNOWN:
            if length:
                problems.append(length)
            break
        logger.info("要約の読み上げが %.1f 秒なので、1 回だけ書き直させます: %s", sec or 0.0, r.id)
        retry = cfg.summarize_retry_short if length == LENGTH_REASON_LONG else cfg.summarize_retry_long
        messages += [{"role": "assistant", "content": summary},
                     {"role": "user", "content": retry.format(sec=sec, chars=target_chars(summary, sec or 0.0, cfg))}]
    r.summary = summary
    r.summary_checks = {"result": "held" if problems else "ok", "reasons": problems, "speech_sec": sec,
                        "checked_at": datetime.now(JST).isoformat(timespec="seconds"), "by": "night"}
    r.summarized_hash = r.content_hash
    refresh_status(r)
    return True


_DEADLINE_WORDS = ("締切", "締め切り", "申込期限", "申し込み期限", "期限")
_SPAN_WORDS = ("期間", "から", "まで", "～", "〜")


def date_list(r: NewsRecord, today: date, cfg: NewsConfig) -> str:
    """要約の入力に付ける日付の一覧（規則で本文から抜く。実施日・期間・締切と、掲載日は実施日ではないこと）。"""
    from news_collect import body_dates

    published = r.published_date()
    lines: list[str] = []
    for raw in [r.title, *r.body.splitlines()]:
        line = re.sub(r"\s+", " ", raw.replace("|", " ")).strip()
        found = body_dates(line, published) if line else []
        if not found:
            continue
        kind = ("締切" if any(w in line for w in _DEADLINE_WORDS)
                else "開始（続いている）" if len(found) == 1 and re.search(r"日\S{0,6}(から|より|以降)", line)
                else "期間" if any(w in line for w in _SPAN_WORDS) else "実施日など")
        past = "（今日より前）" if all(d < today for d in found) else ""
        days = "・".join(f"{d.month}月{d.day}日" for d in found)  # 年は書かない（本文の和暦と西暦の食い違いを作らない）
        entry = f"- {kind}: {days}{past}　{line[:cfg.date_list_line_chars]}"
        if entry not in lines:
            lines.append(entry)
        if len(lines) >= cfg.date_list_max:
            break
    head = cfg.published_note.format(month=published.month, day=published.day) if published != date.min else ""
    return "\n".join([*(lines or [cfg.date_list_none]), *([head] if head else [])])


def measure(meter: SpeechMeter | None, text: str) -> float | None:
    if meter is None or not text:
        return None
    try:
        return meter.seconds(text)
    except (RuntimeError, httpx.HTTPError, wave.Error, EOFError):
        logger.warning("読み上げの長さを測れません（VOICEVOX）", exc_info=True)
        return None


def recheck_staff_summary(r: NewsRecord, checker: Dialogue, today: date) -> None:
    """職員が直した要約は書き直さず、本文が変わったら安全装置だけ掛け直す。"""
    problems = checks_for(r.summary, r, checker, today)
    r.summary_checks = {**r.summary_checks, "result": "held" if problems else "ok", "reasons": problems,
                        "checked_at": datetime.now(JST).isoformat(timespec="seconds"), "by": "night"}
    r.summarized_hash = r.content_hash
    refresh_status(r)


def english_only_hold(r: NewsRecord) -> bool:
    """英字だけが理由で保留になった要約か（WP13 で英字の規則を「大文字の略語は 3 語まで」に緩めた。前の理由は「英単語」）。"""
    reasons = [str(x) for x in (r.summary_checks or {}).get("reasons") or []]
    return (bool(r.summary) and (r.summary_checks or {}).get("result") == "held" and bool(reasons)
            and all(x == "英単語" or x.startswith("英字") for x in reasons))


def recheck_english_hold(r: NewsRecord, checker: Dialogue, today: date) -> None:
    """英字だけで保留の要約を、書き直さずに今の規則で掛け直す（読み上げの長さは前に測ったまま）。"""
    problems = checks_for(r.summary, r, checker, today)
    r.summary_checks = {**r.summary_checks, "result": "held" if problems else "ok", "reasons": problems,
                        "checked_at": datetime.now(JST).isoformat(timespec="seconds"), "by": "night"}
    refresh_status(r)


def summarize(store: Path, site: str, llm: NightLLM, meter: SpeechMeter | None, cfg: NewsConfig | None = None,
              force: bool = False, now: datetime | None = None) -> dict[str, int]:
    cfg = cfg or config.news
    at = now or datetime.now(JST)
    checker = make_checker(site, at)
    stats = {"summarized": 0, "ok": 0, "held": 0, "pending": 0, "staff": 0, "expired": 0, "tidied": 0, "rejudged": 0}
    for r in load_records(store):
        tidy = tidy_summary(r.summary)
        if tidy != r.summary:  # 前に保存した要約にも後処理を当てる（「10時00分」→「10時」）
            r.summary = tidy
            save_record(store, r)
            stats["tidied"] += 1
        if r.relevance != "introduce":
            continue
        if r.expired(at.date()):
            stats["expired"] += 1
            continue
        if r.edited_by_staff:
            if r.summarized_hash != r.content_hash:
                recheck_staff_summary(r, checker, at.date())
                save_record(store, r)
            stats["staff"] += 1
        elif not force and r.summarized_hash == r.content_hash and english_only_hold(r):
            # 英字だけが理由の保留（職員が直したものは上で触らない）: 今の規則で掛け直す
            recheck_english_hold(r, checker, at.date())
            save_record(store, r)
            stats["rejudged"] += 1
        elif force or r.summarized_hash != r.content_hash or not r.summary:
            if not summarize_record(r, llm, meter, checker, cfg, at.date()):
                stats["pending"] += 1
                continue
            save_record(store, r)
            stats["summarized"] += 1
        stats["held" if r.held else "ok"] += 1
    logger.info("お知らせの要約: %s", stats)
    return stats


def main() -> int:
    import build
    from news_collect import store_dir

    ap = argparse.ArgumentParser(description="お知らせの要約（WP11）")
    ap.add_argument("--site", required=True)
    ap.add_argument("--force", action="store_true", help="要約済みも作り直す（職員が直したものは除く）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    llm, meter = NightLLM(), SpeechMeter()
    try:
        summarize(store_dir(build.data_dir() / "kb", args.site), args.site, llm, meter, force=args.force)
    finally:
        llm.close()
        meter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
