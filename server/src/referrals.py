"""紹介状・来訪記録・職員パネル用の集計（WP04 B）。

会話由来でディスクに残すのは、ここに書く 2 つだけ（CLAUDE.md の例外）:

- 紹介状 ``$DATA/referrals/<site>/<date>/<id>.json``: 本人が「伝えて」と同意したものだけ。
  氏名・電話番号は本人が言った場合のみ。保持は referral.retention_days（既定 30 日）
- 来訪記録 ``$DATA/visits/<site>/<date>.jsonl``: 1 会話 1 行の匿名の記録（本文・氏名・音声は含めない）。保持 1 年

安芸太田は kintone 取り込み用に、同じ内容を 1 日 1 ファイルの CSV（UTF-8 BOM、ヘッダ固定）にも書く。
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
import shutil
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from .knowledge import JST

logger = logging.getLogger(__name__)

STATUSES = ("未対応", "対応中", "完了")
RESOLUTIONS = ("完了", "来庁待ち", "電話済み", "取り下げ", "不明")
HANDLER_ROLES = ("窓口", "担当課")
CATEGORIES = ("手続き", "相談", "意見", "その他")
URGENCIES = ("急ぐ", "通常")
CONTACT_PREFS = ("窓口で待つ", "電話", "後日来庁", "未定")
VISIT_RETENTION_DAYS = 365
# 担当課が規則で決まらなかった紹介状の課（職員が職員パネルで付ける）
UNSPECIFIED_DEPARTMENT = "未特定"
CSV_HEADER = ("受付番号", "受付日時", "拠点", "担当課", "区分", "要約", "急ぎ", "連絡方法", "氏名", "電話",
              "要対応", "タグ", "話題", "状態", "対応日時", "対応者", "結果", "id", "地区", "聞き取り", "案内役のメモ")


def load_tags(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [str(t) for t in data.get("tags") or []]


def pick_tags(raw: Any, vocabulary: list[str], limit: int = 5) -> list[str]:
    """LLM が選んだタグを統制語彙に絞る（語彙に無いものは捨てる）。"""
    items = raw if isinstance(raw, list) else []
    return [t for t in dict.fromkeys(str(x) for x in items) if t in vocabulary][:limit]


# 区切りは「-」「の」「、」空白など（「090の1234の5678」と言われることが多い）
_PHONE_RE = re.compile(r"(0\d{1,4})[-ー－‐\s、,の]{0,2}(\d{1,4})[-ー－‐\s、,の]{0,2}(\d{3,4})")


# 全角数字と漢数字（一桁ずつ）を半角に
_DIGITS = str.maketrans("０１２３４５６７８９〇零一二三四五六七八九", "0123456789" "00" "123456789")


# 電話番号の桁数（固定電話 10 桁・携帯 11 桁）。区切って言われた断片はこの桁数まで連結してから確定する
PHONE_DIGITS_MIN = 10
PHONE_DIGITS_MAX = 11


def normalize_digits(text: str) -> str:
    """全角数字・漢数字（一桁ずつ）・「ゼロ」「マル」を半角の数字に。"""
    t = text.translate(_DIGITS)
    # 一桁ずつ読んだ番号は ASR が「ゼロ九ゼロの一二三四…」と書く
    return re.sub(r"ゼロ|ぜろ|マル|まる", "0", t)


def find_phone(text: str) -> str | None:
    """来訪者の発話の中の電話番号（全角も）。言っていなければ None。

    区切りの途中まで（「090の1234の」）を番号とみなさないよう、桁数が 10〜11 のものだけを返す。
    """
    m = _PHONE_RE.search(normalize_digits(text))
    if m is None or not PHONE_DIGITS_MIN <= len("".join(m.groups())) <= PHONE_DIGITS_MAX:
        return None
    return "-".join(m.groups())


def phone_digits(text: str) -> str:
    """発話の中の数字だけを並べたもの（ゼロ・漢数字・全角を半角に。「の」「ー」などの区切りは捨てる）。"""
    return re.sub(r"\D", "", normalize_digits(text))


def format_phone(digits: str) -> str | None:
    """連結した数字を「090-1234-5678」の形に。桁数が電話番号でなければ None。"""
    if not PHONE_DIGITS_MIN <= len(digits) <= PHONE_DIGITS_MAX or not digits.startswith("0"):
        return None
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if digits[1] in "36":  # 東京・大阪（03・06）は 2-4-4
        return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"  # 市外局番 4 桁（江田島 0823・安芸太田 0826）


def phone_for_speech(phone: str) -> str:
    """復唱用（数字を 1 つずつ、区切りに読点）。"""
    return "、".join(" ".join(part) for part in phone.split("-"))


@dataclass
class Referral:
    id: str
    site: str
    created_at: str
    receipt: str
    department: str
    category: str
    summary: str
    urgency: str
    contact_pref: str
    name: str | None = None
    phone: str | None = None
    language: str = "ja"
    edge_id: str = ""
    # 分析用（WP04 B-1 追記）
    needs_action: bool = True
    tags: list[str] = field(default_factory=list)
    topic_free: str = ""
    # 同意の後の聞き取り（質問と答えの組）と、お住まいの地区（番地は転入・転居のときだけ）
    intake: list[dict[str, str]] = field(default_factory=list)
    address_area: str | None = None
    # 案内役が提案・案内した内容（要約には入れない。職員の参考に 1 行）
    assistant_note: str = ""
    # 職員パネルで書く
    status: str = "未対応"
    handled_at: str | None = None
    handled_by_role: str | None = None
    resolution: str | None = None


class ReferralStore:
    """紹介状の保存・一覧・状態の更新・期限での削除。"""

    def __init__(self, data_dir: Path, site: str, retention_days: int, csv_export: bool) -> None:
        self.root = data_dir / "referrals" / site
        self.site = site
        self.retention_days = retention_days
        self.csv_export = csv_export

    def _day_dir(self, d: date) -> Path:
        return self.root / d.isoformat()

    def new_receipt(self, d: date) -> str:
        """その日に使っていない 4 桁の受付番号。"""
        used = {r.receipt for r in self.list(d)}
        while True:
            n = f"{random.randint(1000, 9999)}"
            if n not in used:
                return n

    def create(self, fields: dict[str, Any], edge_id: str = "", language: str = "ja") -> Referral:
        now = datetime.now(JST)
        ref = Referral(
            id=uuid.uuid4().hex[:12], site=self.site, created_at=now.isoformat(timespec="seconds"),
            receipt=self.new_receipt(now.date()), edge_id=edge_id, language=language, **fields,
        )
        self.save(ref)
        return ref

    def save(self, ref: Referral) -> None:
        d = date.fromisoformat(ref.created_at[:10])
        day = self._day_dir(d)
        day.mkdir(parents=True, exist_ok=True)
        tmp = day / f".{ref.id}.tmp"
        tmp.write_text(json.dumps(asdict(ref), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(day / f"{ref.id}.json")
        if self.csv_export:
            self._write_csv(d)

    def list(self, d: date) -> list[Referral]:
        day = self._day_dir(d)
        if not day.is_dir():
            return []
        out: list[Referral] = []
        for f in sorted(day.glob("*.json")):
            try:
                out.append(Referral(**json.loads(f.read_text(encoding="utf-8"))))
            except (OSError, TypeError, json.JSONDecodeError):
                logger.warning("紹介状を読めません: %s", f)
        return sorted(out, key=lambda r: r.created_at)

    def find(self, ref_id: str, d: date) -> Referral | None:
        return next((r for r in self.list(d) if r.id == ref_id), None)

    def update(self, ref_id: str, d: date, *, status: str | None = None, needs_action: bool | None = None,
               handled_by_role: str | None = None, resolution: str | None = None,
               department: str | None = None, departments: tuple[str, ...] = ()) -> Referral | None:
        """職員パネルからの更新。状態を変えたら対応日時を書く（受付から完了までの時間を出すため）。

        department は departments（拠点の課の一覧）のどれかのときだけ受け付ける。
        """
        ref = self.find(ref_id, d)
        if ref is None:
            return None
        if department is not None and department in departments:
            ref.department = department
        if status is not None and status in STATUSES and status != ref.status:
            ref.status = status
            ref.handled_at = datetime.now(JST).isoformat(timespec="seconds")
        if needs_action is not None:
            ref.needs_action = needs_action
        if handled_by_role is not None and handled_by_role in HANDLER_ROLES:
            ref.handled_by_role = handled_by_role
        if resolution is not None and resolution in RESOLUTIONS:
            ref.resolution = resolution
        self.save(ref)
        return ref

    def _write_csv(self, d: date) -> None:
        path = self.root / f"{d.isoformat()}.csv"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            for r in self.list(d):
                w.writerow([r.receipt, r.created_at, r.site, r.department, r.category, r.summary, r.urgency,
                            r.contact_pref, r.name or "", r.phone or "", "要" if r.needs_action else "不要",
                            "／".join(r.tags), r.topic_free, r.status, r.handled_at or "",
                            r.handled_by_role or "", r.resolution or "", r.id, r.address_area or "",
                            "／".join(f"{q.get('question', '')}: {q.get('answer', '')}" for q in r.intake),
                            r.assistant_note])
        tmp.replace(path)

    def cleanup(self, today: date | None = None) -> int:
        """保持期間を過ぎた日の紹介状（と CSV）を消す。消した日数を返す。"""
        today = today or datetime.now(JST).date()
        return _remove_older(self.root, today - timedelta(days=self.retention_days))


class VisitLog:
    """匿名の来訪記録（1 会話 1 行）。本文・氏名・音声は書かない。"""

    def __init__(self, data_dir: Path, site: str) -> None:
        self.root = data_dir / "visits" / site

    def write(self, record: dict[str, Any]) -> None:
        d = record.get("started_at", "")[:10] or datetime.now(JST).date().isoformat()
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f"{d}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, d: date) -> list[dict[str, Any]]:
        path = self.root / f"{d.isoformat()}.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def cleanup(self, today: date | None = None) -> int:
        today = today or datetime.now(JST).date()
        return _remove_older(self.root, today - timedelta(days=VISIT_RETENTION_DAYS))


class TidbitStore:
    """匿名の小ネタ（本人が同意し、読み上げて確かめた一言だけ）。1 行 1 件、期限で消す。"""

    def __init__(self, data_dir: Path, site: str, retention_days: int) -> None:
        self.path = data_dir / "tidbits" / f"{site}.jsonl"
        self.retention_days = retention_days

    def add(self, text: str) -> dict[str, Any]:
        item = {"id": uuid.uuid4().hex[:12],
                "created_at": datetime.now(JST).isoformat(timespec="seconds"), "text": text}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def active(self) -> list[dict[str, Any]]:
        """期限内のもの（新しい順）。"""
        keep_from = datetime.now(JST) - timedelta(days=self.retention_days)
        return [t for t in reversed(self._read()) if _parse_time(t.get("created_at")) >= keep_from]

    def delete(self, tid: str) -> bool:
        items = self._read()
        rest = [t for t in items if t.get("id") != tid]
        if len(rest) == len(items):
            return False
        self._write(rest)
        return True

    def cleanup(self) -> int:
        items = self._read()
        keep_from = datetime.now(JST) - timedelta(days=self.retention_days)
        rest = [t for t in items if _parse_time(t.get("created_at")) >= keep_from]
        if len(rest) != len(items):
            self._write(rest)
        return len(items) - len(rest)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                out.append(item)
        return out

    def _write(self, items: list[dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in items), encoding="utf-8")
        tmp.replace(self.path)


def _parse_time(value: Any) -> datetime:
    try:
        t = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=JST)
    return t if t.tzinfo else t.replace(tzinfo=JST)


def _remove_older(root: Path, keep_from: date) -> int:
    """root の下の「日付」名のディレクトリ・ファイル（YYYY-MM-DD[.csv/.jsonl]）で keep_from より前を消す。"""
    if not root.is_dir():
        return 0
    removed = 0
    for p in root.iterdir():
        try:
            d = date.fromisoformat(p.name[:10])
        except ValueError:
            continue
        if d < keep_from:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed += 1
    return removed


def today_stats(store: ReferralStore, visits: VisitLog, d: date) -> dict[str, Any]:
    """職員パネルの「今日の状況」（個人を特定する情報は含めない）。"""
    refs = store.list(d)
    vs = visits.read(d)
    tags = Counter(t for v in vs for t in v.get("tags") or [])
    tags.update(t for r in refs if not any(v.get("referral_id") == r.id for v in vs) for t in r.tags)
    return {
        "date": d.isoformat(),
        "visits": len(vs),
        "referrals": len(refs),
        "pending_action": sum(1 for r in refs if r.needs_action and r.status != "完了"),
        "unspecified": sum(1 for r in refs if r.department in ("", UNSPECIFIED_DEPARTMENT)),
        "languages": dict(Counter(v.get("language", "ja") for v in vs)),
        "top_tags": tags.most_common(5),
    }
