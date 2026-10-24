"""職員パネルの「お知らせ」タブの中身（WP11 B-1）。

一覧（題名・掲載日・分類・状態・期限・紹介した回数・画像の有無）、要約の編集（保存の前に夜と同じ安全装置を通し、
通らなければ理由を返して保存しない）、止める／自動に戻す／優先にする、画像を出さない。
書き込みは記事の JSON を直接更新する（git 管理外。同時編集は考えない）。フロントの読み直しを待たずに次の紹介から効く。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from .config import DialogueConfig, NewsConfig
from .knowledge import JST
from .news import (
    STATUS_LABELS,
    NewsBook,
    NewsRecord,
    TextChecker,
    is_candidate,
    read_fetch_status,
    record_to_dict,
    refresh_status,
    summary_problems,
)

logger = logging.getLogger(__name__)

ACTIONS = ("stop", "auto", "priority", "hide_image", "show_image")


class SummaryRejected(ValueError):
    """要約が安全装置を通らなかった（理由の一覧を持つ）。"""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("、".join(reasons))
        self.reasons = reasons


class NewsPanel:
    def __init__(self, book: NewsBook, checker: Callable[[], TextChecker], dialogue_cfg: DialogueConfig,
                 cfg: NewsConfig, today: Callable[[], date] = lambda: datetime.now(JST).date(),
                 speech_seconds: Callable[[str], float] | None = None) -> None:
        self.book = book
        self.checker = checker
        self.cfg = cfg
        self.today = today
        # 読み上げの長さ（秒）を測る（VOICEVOX。「確認済みとして保存」の長さの検査に使う。無ければ字数だけ）
        self.speech_seconds = speech_seconds
        self._claim_re = re.compile(dialogue_cfg.grounding_claim_pattern)
        self._future_re = re.compile(cfg.summary_future_pattern)

    def row(self, r: NewsRecord, today: date) -> dict[str, Any]:
        return {"id": r.id, "title": r.title, "published": r.published, "category": r.category,
                "status": r.status, "status_label": STATUS_LABELS.get(r.status, r.status),
                "relevance": r.relevance, "expires_at": r.expires_at, "expired": r.expired(today),
                "shown_count": r.shown_count, "has_image": bool(r.image), "image_hidden": r.image_hidden,
                "candidate": is_candidate(r, today), "reasons": r.summary_checks.get("reasons", []),
                "edited_by_staff": r.edited_by_staff, "verified": r.verified, "verified_stale": r.verified_stale}

    def rows(self, status: str | None = None, category: str | None = None,
             include_skip: bool = False) -> list[dict[str, Any]]:
        """一覧。既定は紹介する分類（introduce）の記事だけ（紹介しない記事は include_skip で）。"""
        self.book.reload(force=True)
        today = self.today()
        records = [r for r in self.book.records() if include_skip or r.relevance == "introduce"]
        if status:
            records = [r for r in records if r.status == status]
        if category:
            records = [r for r in records if r.category == category]
        order = {r.id: i for i, r in enumerate(self.book.candidates())}
        # 紹介の順番（候補）→ 保留 → それ以外（掲載日の新しい順）
        records.sort(key=lambda r: (order.get(r.id, len(order) + (0 if r.status == "held" else 1)),
                                    -r.published_date().toordinal()))
        return [self.row(r, today) for r in records]

    def counts(self) -> dict[str, int]:
        today = self.today()
        intro = [r for r in self.book.records() if r.relevance == "introduce"]
        return {"introduce": len(intro), "candidates": sum(is_candidate(r, today) for r in intro),
                "held": sum(r.status == "held" for r in intro), "stopped": sum(r.status == "stopped" for r in intro),
                "priority": sum(r.status == "priority" for r in intro), "skip": sum(
                    r.relevance == "skip" for r in self.book.records())}

    def fetch_status(self) -> dict[str, Any] | None:
        """夜の収集の最後の結果（職員パネルの上に「最終取得: M月D日 HH:MM（成功／失敗: 理由）」）。"""
        st = read_fetch_status(self.book.directory)
        if st is None:
            return None
        try:
            at = datetime.fromisoformat(str(st.get("at")))
        except ValueError:
            return None
        ok = bool(st.get("ok"))
        label = f"最終取得: {at.month}月{at.day}日 {at:%H:%M}（{'成功' if ok else '失敗: ' + str(st.get('reason') or '')}）"
        return {"ok": ok, "at": st.get("at"), "label": label, "reason": st.get("reason", "")}

    def detail(self, rid: str) -> dict[str, Any] | None:
        r = self.book.get(rid)
        if r is None:
            return None
        d = record_to_dict(r)
        d["status_label"] = STATUS_LABELS.get(r.status, r.status)
        d["candidate"] = is_candidate(r, self.today())
        return d

    def _get(self, rid: str) -> NewsRecord:
        r = self.book.get(rid)
        if r is None:
            raise KeyError(rid)
        return r

    def check(self, rid: str, summary: str) -> list[str]:
        """要約の安全装置（夜の要約と同じ。読み上げの長さは夜だけ測る）。"""
        return summary_problems(summary, self._get(rid), self.checker(), self.today(), self._claim_re,
                                self._future_re)

    def length_problems(self, text: str) -> tuple[list[str], float | None]:
        """長さの検査（字数と読み上げの秒数。夜の要約と同じ範囲）。"""
        cfg = self.cfg
        problems: list[str] = []
        n = len(re.sub(r"\s", "", text))
        if not cfg.summary_min_chars <= n <= cfg.summary_max_chars:
            problems.append(f"長さ（{n} 字。{cfg.summary_min_chars}〜{cfg.summary_max_chars} 字）")
        sec: float | None = None
        if self.speech_seconds is not None:
            try:
                sec = self.speech_seconds(text)
            except RuntimeError:
                problems.append("読み上げの長さを測れません（VOICEVOX）")
        if sec is not None and not cfg.speech_min_sec <= sec <= cfg.speech_max_sec:
            problems.append(f"読み上げ {sec} 秒（{cfg.speech_min_sec:g}〜{cfg.speech_max_sec:g} 秒）")
        return problems, sec

    def edit_summary(self, rid: str, summary: str, by: str, verified: bool = False) -> dict[str, Any]:
        """要約を直す。安全装置に引っかかる文は保存しない（SummaryRejected に理由）。

        verified（「確認済みとして保存」）: 職員が記事と照らして確かめたものとして、日付・期限・金額・支払方法の本文照合だけを外し、
        長さ（字数・読み上げの秒数）を足して見る。手続きナレッジの「承認」と同じく、人が確かめたものは機械の照合より上に置く。
        """
        r = self._get(rid)
        text = re.sub(r"\s*\n\s*", "", summary.strip())
        problems = self.check(rid, text)
        sec: float | None = None
        skipped: list[str] = []
        if verified:
            skipped = [p for p in problems if p in self.cfg.verified_skip_reasons]
            problems = [p for p in problems if p not in self.cfg.verified_skip_reasons]
            length, sec = self.length_problems(text)
            problems += length
        if problems:
            raise SummaryRejected(problems)
        r.summary = text
        r.edited_by_staff, r.edited_by = True, by
        now = datetime.now(JST).isoformat(timespec="seconds")
        r.summary_checks = {"result": "ok", "reasons": [], "speech_sec": sec, "by": "staff_verified" if verified else "staff",
                            "checked_at": now, **({"skipped": skipped} if skipped else {})}
        if verified:
            r.verified_by, r.verified_at, r.verified_hash = by, now, r.content_hash
        else:
            r.verified_by, r.verified_at, r.verified_hash = "", "", ""  # ふつうに直したら確認済みの印は外す
        r.summarized_hash = r.content_hash
        refresh_status(r)
        self.book.save(r)
        logger.info("職員パネルでお知らせの要約を直しました: %s", rid)
        out = self.detail(rid)
        assert out is not None
        return out

    def act(self, rid: str, action: str) -> dict[str, Any]:
        """止める（stop）／自動に戻す（auto）／優先にする（priority）／画像を出さない（hide_image）・出す（show_image）。"""
        if action not in ACTIONS:
            raise ValueError(f"操作は {', '.join(ACTIONS)} のどれかです: {action}")
        r = self._get(rid)
        if action == "stop":
            r.status = "stopped"
        elif action == "priority":
            r.status = "priority"
        elif action == "auto":
            r.status = "auto"
            refresh_status(r)  # 要約が通っていなければ保留のまま
        else:
            r.image_hidden = action == "hide_image"
        self.book.save(r)
        logger.info("職員パネルでお知らせを操作しました: %s %s", rid, action)
        out = self.detail(rid)
        assert out is not None
        return out
