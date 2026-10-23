"""職員パネルの「手続き」タブの中身（WP09 B-2）。

一覧・詳細の編集・承認・差し戻し・要再確認の解除と、「想定問答を試す」（その手続きを承認済みとみなして、
faq の問いを実際の対話に通す）。×を付けた答えは ``$DATA/review/procedures_<date>.jsonl`` に素材として書く。
書き込みは手続きナレッジの YAML を直接更新する（git 管理外。同時編集は考えない）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .knowledge import JST
from .procedure_kb import (
    PROVISIONAL_BY,
    STATUS_LABELS,
    ProcedureBook,
    ProcedureRecord,
    record_from_dict,
    record_to_dict,
)

logger = logging.getLogger(__name__)

# 職員パネルから書き換えてよい項目（状態・承認の記録・出典・ハッシュは専用の操作で）
EDITABLE = ("name", "aliases", "when", "department", "floor", "audience", "bring", "steps", "deadline", "fee",
            "related", "intake", "faq", "staff_note", "tags")
REVIEW_SUBDIR = "review"

# faq の問い 1 つを対話に通して答えを返す（main.py が Dialogue を組んで渡す）
FaqRunner = Callable[[ProcedureRecord, str], Awaitable[str]]


def summary_row(r: ProcedureRecord) -> dict[str, Any]:
    checked = max((s.checked_at for s in r.sources if s.checked_at), default="")
    return {"id": r.id, "name": r.name, "department": r.department, "status": r.status,
            "status_label": STATUS_LABELS.get(r.status, r.status), "checked_at": checked,
            "faq": len(r.faq), "sources": len(r.sources), "recheck_reason": r.recheck_reason,
            "approved_by": r.approved_by, "approved_at": r.approved_at, "provisional": r.provisional}


def parse_bring(lines: list[Any]) -> list[Any]:
    """持ち物の入力（1 行 1 つ。「品名 | 語1、語2」で、その語が話に出たときだけ言う持ち物）。"""
    out: list[Any] = []
    for x in lines:
        if isinstance(x, dict):
            out.append(x)
            continue
        item, _, when = str(x).partition("|")
        item = item.strip()
        if not item:
            continue
        words = [w.strip() for w in re.split(r"[、,，]", when) if w.strip()]
        out.append({"item": item, "when": words} if words else item)
    return out


class ProcedurePanel:
    def __init__(self, book: ProcedureBook, data_dir: Path, departments: tuple[str, ...] = (),
                 runner: FaqRunner | None = None, try_timeout_sec: float = 60.0) -> None:
        self.book = book
        self.review_dir = data_dir / REVIEW_SUBDIR
        self.departments = departments
        self.runner = runner
        self.try_timeout_sec = try_timeout_sec

    def rows(self, status: str | None = None) -> list[dict[str, Any]]:
        """一覧。status は draft/approved/recheck のほか provisional（開発者の仮承認のまま）。"""
        records = self.book.records()
        if status == "provisional":
            records = [r for r in records if r.provisional]
        elif status:
            records = [r for r in records if r.status == status]
        rows = [summary_row(r) for r in records]
        # 要再確認を先に（目立たせる）、次に下書き、承認済み
        order = {"recheck": 0, "draft": 1, "approved": 2}
        return sorted(rows, key=lambda x: (order.get(x["status"], 3), x["id"]))

    def detail(self, rid: str) -> dict[str, Any] | None:
        r = self.book.get(rid)
        if r is None:
            return None
        d = record_to_dict(r)
        d["status_label"] = STATUS_LABELS.get(r.status, r.status)
        d["provisional"] = r.provisional
        d["bring_lines"] = [b.item + (" | " + "、".join(b.when) if b.when else "") for b in r.bring]
        return d

    def approve(self, rid: str, by: str) -> None:
        """職員の承認（仮承認のものを押し直すと、職員の名前の通常の承認に置き換わる）。"""
        if not by.strip() or by.strip() == PROVISIONAL_BY:
            raise ValueError("承認する人の名前が要ります")
        self.book.approve(rid, by.strip())

    def edit(self, rid: str, fields: dict[str, Any], by: str) -> dict[str, Any]:
        r = self.book.get(rid)
        if r is None:
            raise KeyError(rid)
        unknown = set(fields) - set(EDITABLE)
        if unknown:
            raise ValueError(f"書き換えられない項目です: {', '.join(sorted(unknown))}")
        if fields.get("department") and self.departments and fields["department"] not in self.departments:
            raise ValueError(f"課は拠点の課から選んでください: {fields['department']}")
        data = record_to_dict(r)
        for k, v in fields.items():
            data[k] = parse_bring(list(v or [])) if k == "bring" else v
        new = record_from_dict(data)
        if new.status == "approved":
            # 承認済みを職員が直したら、直した人が承認したことにする
            new = replace(new, approved_by=by, approved_at=datetime.now(JST).isoformat(timespec="seconds"))
        self.book.save(new)
        logger.info("職員パネルで手続きナレッジを編集しました: %s", rid)
        out = self.detail(rid)
        assert out is not None
        return out

    async def try_faq(self, rid: str) -> list[dict[str, str]]:
        """faq の問いを順に対話に通す（その手続きは承認済みとみなす）。"""
        r = self.book.get(rid)
        if r is None:
            raise KeyError(rid)
        if self.runner is None:
            raise RuntimeError("想定問答を試す仕組みがありません（LLM につながっていない）")
        out = []
        for q, a in r.faq:
            try:
                got = await asyncio.wait_for(self.runner(r, q), self.try_timeout_sec)
            except Exception as e:  # 1 問の失敗で全体を止めない
                logger.warning("想定問答を試せませんでした: %s", rid, exc_info=True)
                got = f"（試せませんでした: {type(e).__name__}）"
            out.append({"q": q, "expected": a, "answer": got})
        return out

    def review(self, rid: str, q: str, answer: str, ok: bool, note: str = "") -> bool:
        """○×を受ける。×だけ誤り探しの素材に書く（返り値は書いたか）。"""
        if ok:
            return False
        now = datetime.now(JST)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        path = self.review_dir / f"procedures_{now.date().isoformat()}.jsonl"
        item = {"kind": "procedure_faq", "procedure": rid, "q": q, "answer": answer, "note": note,
                "at": now.isoformat(timespec="seconds")}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("想定問答の×を誤り探しの素材に書きました: %s", rid)
        return True


def approved_view(record: ProcedureRecord, base: ProcedureBook) -> tuple[ProcedureBook, tempfile.TemporaryDirectory[str]]:
    """record だけを承認済みにした手続きナレッジ（一時ディレクトリ。想定問答を試すとき用）。"""
    tmp = tempfile.TemporaryDirectory(prefix="idobata-try-")
    book = ProcedureBook(Path(tmp.name), True, base.other_surfaces)
    book.save(replace(record, status="approved", approved_at=record.approved_at or "試し"))
    return book, tmp
