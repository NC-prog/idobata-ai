"""手続きナレッジ（職員が承認する正解表、WP09）。

``~/idobata-data/procedures/<site>/<id>.yaml`` に 1 手続き 1 ファイル。クロールしたパック（市の文章の写し）とは別に、
「この相談はどこへ・何を持って・どう進む」を構造化して持つ。市が自分で育てる知識（git 管理外）。

- 状態は draft（下書き）/ approved（承認済み）/ recheck（要再確認。出典のページが変わった）
- 対話で使うのは承認済みだけ（recheck は、承認済みだったものを設定で使い続けられる）
- 対話での優先順位: 固定事実（facility）＞承認済みの手続きナレッジ＞パック（クロール）＞LLM の一般知識
- ファイルは職員パネルからも書き換わる。読むたびに更新時刻を見て、変わっていれば読み直す（再起動不要）
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .knowledge import JST, Reference

logger = logging.getLogger(__name__)

STATUSES = ("draft", "approved", "recheck")
STATUS_LABELS = {"draft": "下書き", "approved": "承認済み", "recheck": "要再確認"}
# Reference.trust の値（手続きナレッジから作った参考資料の印）
PROCEDURE_TRUST = "procedure"
PROCEDURE_LABEL = "市の案内（承認済み）"
_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
# 名前の括弧書き（「国民健康保険の加入（退職・転入など）」）は発話との突き合わせに使わない
_PAREN_RE = re.compile(r"[（(].*?[）)]")


@dataclass(frozen=True)
class BringItem:
    """持ち物の案内 1 つ。when の語が聞き取りの答えか発話にあれば言う（空なら常に）。"""

    item: str
    when: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntakeSet:
    """紹介状の同意の後に伺う項目の組と、持ち物。"""

    questions: tuple[str, ...]
    when: tuple[str, ...] = ()
    bring: tuple[BringItem, ...] = ()

    def bring_for(self, text: str) -> str | None:
        """聞き取りの答えと発話から分かる持ち物（最初に合った 1 つ）。"""
        for b in self.bring:
            if not b.when or any(w in text for w in b.when):
                return b.item
        return None


@dataclass(frozen=True)
class Source:
    url: str
    checked_at: str = ""
    # このページを根拠にした項目（bring / steps / deadline / fee / audience / faq）
    items: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcedureRecord:
    """手続き 1 件（docs/WP09_procedure_knowledge.md の A-1 の形式）。"""

    id: str
    name: str
    aliases: tuple[str, ...] = ()
    department: str = ""
    floor: str = ""
    audience: str = ""
    bring: tuple[BringItem, ...] = ()
    steps: tuple[str, ...] = ()
    deadline: str | None = None
    fee: str | None = None
    related: tuple[str, ...] = ()
    # 紹介状の聞き取り（key, question）
    intake: tuple[tuple[str, str], ...] = ()
    faq: tuple[tuple[str, str], ...] = ()
    sources: tuple[Source, ...] = ()
    status: str = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    staff_note: str | None = None
    content_hash: str | None = None
    # 名前・別名に加えて、発話にこの語のどれかがあるときだけ当たる（「マイナンバーカード」＋「なくし」）
    when: tuple[str, ...] = ()
    # 紹介状のタグ（tags.yaml の統制語彙）
    tags: tuple[str, ...] = ()
    # 要再確認になった理由（出典のページが変わった・パックから消えた）
    recheck_reason: str | None = None

    @property
    def surfaces(self) -> tuple[str, ...]:
        """発話と突き合わせる語（名前の括弧書きを除いたものと別名）。"""
        name = _PAREN_RE.sub("", self.name).strip()
        return tuple(dict.fromkeys(s for s in (name, *self.aliases) if len(s) >= 2))

    def usable(self, keep_recheck: bool) -> bool:
        """対話で使ってよいか（承認済み。要再確認は、承認済みだったものを設定で使い続ける）。"""
        if self.status == "approved":
            return True
        return self.status == "recheck" and keep_recheck and bool(self.approved_at)

    def intake_set(self) -> IntakeSet:
        return IntakeSet(tuple(q for _, q in self.intake), (), self.bring)

    def facts_text(self, department_line: str = "") -> str:
        """【市の案内（承認済み）】に入れる本文。"""
        lines = [f"手続き: {self.name}"]
        if department_line or self.department:
            lines.append(f"担当: {department_line or self.department}")
        if self.audience:
            lines.append(f"対象: {self.audience}")
        if self.bring:
            # 条件つきの持ち物は条件も書く（退職なら資格喪失証明書、転入なら転出証明書。両方を並べて言わせない）
            lines.append("持ち物: " + "、".join(
                b.item + (f"（{'・'.join(b.when)} の場合）" if b.when else "") for b in self.bring))
        if self.steps:
            lines.append("流れ: " + " → ".join(self.steps))
        if self.deadline:
            lines.append(f"期限: {self.deadline}")
        if self.fee:
            lines.append(f"手数料・金額: {self.fee}")
        for q, a in self.faq:
            lines.append(f"よくある質問: {q} → {a}")
        if self.staff_note:
            lines.append(f"職員から: {self.staff_note}")
        return "\n".join(lines)

    def to_reference(self, department_line: str = "") -> Reference:
        url = self.sources[0].url if self.sources else ""
        return Reference(self.name, url, self.facts_text(department_line), 1.0, 0.0,
                         trust=PROCEDURE_TRUST, label=PROCEDURE_LABEL, chunk_id=f"procedure:{self.id}")

    def blank_fields(self) -> list[str]:
        """空欄の項目（下書きの要約表に使う）。"""
        out = []
        for key, value in (("bring", self.bring), ("steps", self.steps), ("deadline", self.deadline),
                           ("fee", self.fee), ("audience", self.audience), ("faq", self.faq),
                           ("sources", self.sources), ("department", self.department)):
            if not value:
                out.append(key)
        return out


def _strs(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(x).strip() for x in value if x is not None and str(x).strip())


def _opt(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("null", "none") else None


def record_from_dict(data: dict[str, Any]) -> ProcedureRecord:
    """YAML の 1 件を読む。必須は id と name。"""
    rid, name = str(data.get("id") or "").strip(), str(data.get("name") or "").strip()
    if not _ID_RE.match(rid) or not name:
        raise ValueError(f"id（英小文字・数字・_）と name が要ります: {rid!r}")
    bring: list[BringItem] = []
    for b in data.get("bring") or []:
        if isinstance(b, dict) and b.get("item"):
            bring.append(BringItem(str(b["item"]).strip(), _strs(b.get("when"))))
        elif isinstance(b, str) and b.strip():
            bring.append(BringItem(b.strip()))
    intake = tuple((str(q.get("key") or f"q{i + 1}"), str(q["question"]).strip())
                   for i, q in enumerate(data.get("intake") or [])
                   if isinstance(q, dict) and q.get("question"))
    faq = tuple((str(f["q"]).strip(), str(f["a"]).strip()) for f in data.get("faq") or []
                if isinstance(f, dict) and f.get("q") and f.get("a"))
    sources = tuple(Source(str(s["url"]).strip(), str(s.get("checked_at") or ""), _strs(s.get("items")))
                    for s in data.get("sources") or [] if isinstance(s, dict) and s.get("url"))
    status = str(data.get("status") or "draft").strip()
    if status not in STATUSES:
        raise ValueError(f"status は {'/'.join(STATUSES)}: {status!r}")
    return ProcedureRecord(
        rid, name, _strs(data.get("aliases")), str(data.get("department") or "").strip(),
        str(data.get("floor") or "").strip(), str(data.get("audience") or "").strip(), tuple(bring),
        _strs(data.get("steps")), _opt(data.get("deadline")), _opt(data.get("fee")), _strs(data.get("related")),
        intake, faq, sources, status, _opt(data.get("approved_by")), _opt(data.get("approved_at")),
        _opt(data.get("staff_note")), _opt(data.get("content_hash")), _strs(data.get("when")),
        _strs(data.get("tags")), _opt(data.get("recheck_reason")),
    )


def record_to_dict(r: ProcedureRecord) -> dict[str, Any]:
    """YAML に書く形（A-1 の項目順）。"""
    return {
        "id": r.id, "name": r.name, "aliases": list(r.aliases),
        "when": list(r.when) or None,
        "department": r.department or None, "floor": r.floor or None, "audience": r.audience or None,
        "bring": [{"item": b.item, "when": list(b.when)} if b.when else b.item for b in r.bring],
        "steps": list(r.steps), "deadline": r.deadline, "fee": r.fee, "related": list(r.related),
        "intake": [{"key": k, "question": q} for k, q in r.intake],
        "faq": [{"q": q, "a": a} for q, a in r.faq],
        "sources": [{"url": s.url, "checked_at": s.checked_at, **({"items": list(s.items)} if s.items else {})}
                    for s in r.sources],
        "tags": list(r.tags) or None,
        "status": r.status, "approved_by": r.approved_by, "approved_at": r.approved_at,
        "staff_note": r.staff_note, "content_hash": r.content_hash, "recheck_reason": r.recheck_reason,
    }


def dump_record(r: ProcedureRecord) -> str:
    data = {k: v for k, v in record_to_dict(r).items() if v is not None or k in (
        "deadline", "fee", "approved_by", "approved_at", "staff_note", "content_hash")}
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120)


def content_hash(texts: Iterable[str]) -> str:
    """出典ページの本文のハッシュ（再確認の検知用）。"""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


@dataclass
class _Hit:
    record: ProcedureRecord
    start: int
    end: int
    by_when: bool = False


@dataclass
class ProcedureBook:
    """1 拠点の手続きナレッジ。ファイルの更新時刻が変われば読み直す。"""

    directory: Path
    keep_recheck: bool = True
    # 発話との突き合わせで、これより長い語に含まれる当たりを捨てるための、ほかの語（procedures.yaml の別名表）
    other_surfaces: tuple[str, ...] = ()
    _records: dict[str, ProcedureRecord] = field(default_factory=dict)
    _signature: tuple[tuple[str, float], ...] | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def _scan(self) -> tuple[tuple[str, float], ...]:
        try:
            return tuple(sorted((p.name, p.stat().st_mtime) for p in self.directory.glob("*.yaml")))
        except OSError:
            return ()

    def reload_if_changed(self) -> bool:
        sig = self._scan()
        if sig == self._signature:
            return False
        records: dict[str, ProcedureRecord] = {}
        errors: dict[str, str] = {}
        for name, _ in sig:
            path = self.directory / name
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                r = record_from_dict(data)
            except (OSError, yaml.YAMLError, ValueError, TypeError, AttributeError) as e:
                errors[name] = str(e)
                logger.warning("手続きナレッジを読めません: %s（%s）", path, e)
                continue
            records[r.id] = r
        changed = self._signature is not None
        self._records, self.errors, self._signature = records, errors, sig
        if changed:
            logger.info("手続きナレッジを読み直しました（%d 件）", len(records))
        return True

    def records(self) -> list[ProcedureRecord]:
        self.reload_if_changed()
        return sorted(self._records.values(), key=lambda r: r.id)

    def get(self, rid: str) -> ProcedureRecord | None:
        self.reload_if_changed()
        return self._records.get(rid)

    def usable(self, r: ProcedureRecord) -> bool:
        return r.usable(self.keep_recheck)

    def save(self, r: ProcedureRecord) -> None:
        """1 件を書く（職員パネルの編集・承認、下書きの生成）。"""
        if not _ID_RE.match(r.id):
            raise ValueError(f"id が不正です: {r.id!r}")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{r.id}.yaml"
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(dump_record(r), encoding="utf-8")
        tmp.replace(path)
        self._signature = None  # 次に読むときに読み直す
        self.reload_if_changed()

    def match(self, text: str) -> list[ProcedureRecord]:
        """発話で言った手続き（全状態）。出てきた順。

        - 名前（括弧書きを除く）か別名が発話にあれば当たり。when があれば、その語のどれかも発話に要る
        - 当たった語が、ほかの当たった語（別名表の語も含む）の中に含まれていれば捨てる（「国保」は「国保税」の中）
        - 同じ語で当たった手続きのうち、when で絞った手続きがあれば、when の無いほうは捨てる
        """
        if not text:
            return []
        recs = self.records()
        spans: list[tuple[int, int]] = []
        for s in self.other_surfaces:
            spans += [(m.start(), m.end()) for m in re.finditer(re.escape(s), text)]
        hits: list[_Hit] = []
        for r in recs:
            if r.when and not any(w in text for w in r.when):
                continue
            for s in r.surfaces:
                for m in re.finditer(re.escape(s), text):
                    hits.append(_Hit(r, m.start(), m.end(), bool(r.when)))
                    spans.append((m.start(), m.end()))
        kept = [h for h in hits if not any(
            a <= h.start and h.end <= b and (b - a) > (h.end - h.start) for a, b in spans)]
        out: dict[str, _Hit] = {}
        for h in sorted(kept, key=lambda h: (h.start, -(h.end - h.start))):
            if h.record.id not in out:
                out[h.record.id] = h
        if any(h.by_when for h in out.values()):
            narrowed = {h.start for h in out.values() if h.by_when}
            out = {k: h for k, h in out.items() if h.by_when or h.start not in narrowed}
        return [h.record for h in sorted(out.values(), key=lambda h: h.start)]

    # ---- 職員パネル ----------------------------------------------------

    def approve(self, rid: str, by: str, now: datetime | None = None) -> ProcedureRecord:
        r = self._must(rid)
        stamp = (now or datetime.now(JST)).isoformat(timespec="seconds")
        new = _replace(r, status="approved", approved_by=by, approved_at=stamp, recheck_reason=None)
        self.save(new)
        logger.info("手続きナレッジを承認しました: %s", rid)
        return new

    def send_back(self, rid: str) -> ProcedureRecord:
        """差し戻し（下書きに戻す。承認の記録も消す）。"""
        new = _replace(self._must(rid), status="draft", approved_by=None, approved_at=None)
        self.save(new)
        return new

    def clear_recheck(self, rid: str) -> ProcedureRecord:
        """要再確認を解除（承認済みだったものは承認済みに、それ以外は下書きに戻す）。"""
        r = self._must(rid)
        new = _replace(r, status="approved" if r.approved_at else "draft", recheck_reason=None)
        self.save(new)
        return new

    def mark_recheck(self, rid: str, reason: str, new_hash: str | None) -> ProcedureRecord:
        r = self._must(rid)
        new = _replace(r, status="recheck", recheck_reason=reason, content_hash=new_hash or r.content_hash)
        self.save(new)
        return new

    def _must(self, rid: str) -> ProcedureRecord:
        r = self.get(rid)
        if r is None:
            raise KeyError(rid)
        return r


def _replace(r: ProcedureRecord, **kw: Any) -> ProcedureRecord:
    return replace(r, **kw)


def procedures_dir(data_dir: Path, site: str, subdir: str = "procedures") -> Path:
    return data_dir / subdir / site


def check_sources(book: ProcedureBook, pages: dict[str, list[str]], today: date | None = None) -> list[str]:
    """出典のページの本文（パックのチャンク）のハッシュを比べ、変わっていたら要再確認にする（B-3）。

    pages はパックの source_url → 本文のチャンク（順に）。返り値は要再確認にした手続きの id。
    下書きは要再確認にしない（まだ承認していないので、作り直すときにハッシュを付け直す）。
    """
    del today
    out: list[str] = []
    for r in book.records():
        if not r.sources or r.status == "draft":
            continue
        missing = [s.url for s in r.sources if s.url not in pages]
        texts = [t for s in r.sources for t in pages.get(s.url, [])]
        new_hash = content_hash(texts)
        if missing:
            reason = "出典のページが知識パックにありません: " + "、".join(missing)
        elif r.content_hash and new_hash != r.content_hash:
            reason = "出典のページの本文が変わりました"
        else:
            continue
        if r.status == "recheck" and r.recheck_reason == reason:
            continue
        book.mark_recheck(r.id, reason, None if missing else new_hash)
        logger.info("手続きナレッジを要再確認にしました: %s（%s）", r.id, reason)
        out.append(r.id)
    return out


def load_seeds(path: Path, site: str, now: datetime | None = None) -> list[ProcedureRecord]:
    """procedure_seeds.yaml（WP04 の procedures.yaml から移した手続き）を、拠点の課を入れて承認済みで読む。"""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stamp = (now or datetime.now(JST)).isoformat(timespec="seconds")
    out: list[ProcedureRecord] = []
    for s in data.get("seeds") or []:
        dept = (s.get("department_by_site") or {}).get(site)
        if not dept:
            continue
        d = {k: v for k, v in s.items() if k != "department_by_site"}
        d.update(department=dept, status="approved", approved_by=str(data.get("approved_by") or "移行"),
                 approved_at=stamp)
        out.append(record_from_dict(d))
    return out
