"""会話の文字記録（WP06）。

``$DATA/transcripts/<site>/<date>.jsonl`` に 1 発話 1 行で書く（話者 user / assistant / backchannel）。
音声・映像は残さない。保持は transcript.retention_days（既定 7 日、IDOBATA_TRANSCRIPT_RETENTION_DAYS）で、
起動時に期限を過ぎた日のファイルを消す。セッション単位の「保全」（holds.json）が付いた会話は、
期限を過ぎてもその行だけ残す（保全を外すと次の起動で消える）。

読むのは管理者ロール（/admin/transcripts/、ユーザー admin、IDOBATA_ADMIN_PASSWORD）だけ。
紹介状の聞き取りで伺った氏名・地区・電話番号は書かない（「（お名前を伺いました）」に置き換えてから渡す）。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .knowledge import JST

logger = logging.getLogger(__name__)

SPEAKERS = ("user", "assistant", "backchannel")
HOLDS_FILE = "holds.json"
_DAY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")


class TranscriptStore:
    """文字記録の追記・読み出し・保全・期限での削除。"""

    def __init__(self, data_dir: Path, site: str, retention_days: int) -> None:
        self.root = data_dir / "transcripts" / site
        self.retention_days = retention_days
        self._lock = threading.Lock()

    # ---- 書く ------------------------------------------------------------

    def append(self, speaker: str, session_id: str, edge_id: str, **fields: Any) -> dict[str, Any]:
        """1 発話を 1 行書く。書けなくても会話は止めない（警告だけ）。"""
        if speaker not in SPEAKERS:
            raise ValueError(f"話者は {SPEAKERS} のどれか: {speaker!r}")
        now = datetime.now(JST)
        record = {"ts": now.isoformat(timespec="milliseconds"), "session_id": session_id,
                  "edge_id": edge_id, "speaker": speaker, **fields}
        try:
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                with (self.root / f"{now.date().isoformat()}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("文字記録を書けません", exc_info=True)
        return record

    # ---- 読む ------------------------------------------------------------

    def days(self) -> list[str]:
        """記録のある日（新しい順）。"""
        if not self.root.is_dir():
            return []
        return sorted((m.group(1) for p in self.root.iterdir() if (m := _DAY_FILE_RE.match(p.name))),
                      reverse=True)

    def read(self, d: date) -> list[dict[str, Any]]:
        path = self.root / f"{d.isoformat()}.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out

    def sessions(self, d: date) -> list[dict[str, Any]]:
        """その日のセッションの一覧（始まりの時刻順。行数・保全の有無つき）。"""
        holds = self.holds()
        by_id: dict[str, dict[str, Any]] = {}
        for r in self.read(d):
            sid = str(r.get("session_id") or "")
            s = by_id.setdefault(sid, {"session_id": sid, "started_at": r.get("ts"), "ended_at": r.get("ts"),
                                       "edge_id": r.get("edge_id", ""), "lines": 0, "user_lines": 0})
            s["ended_at"] = r.get("ts")
            s["lines"] += 1
            s["user_lines"] += r.get("speaker") == "user"
        for sid, s in by_id.items():
            s["held"] = sid in holds
        return sorted(by_id.values(), key=lambda s: str(s["started_at"]))

    def session(self, d: date, session_id: str) -> list[dict[str, Any]]:
        return [r for r in self.read(d) if r.get("session_id") == session_id]

    # ---- 保全 ------------------------------------------------------------

    def holds(self) -> dict[str, dict[str, Any]]:
        path = self.root / HOLDS_FILE
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("保全の一覧を読めません: %s", path)
            return {}
        return data if isinstance(data, dict) else {}

    def set_hold(self, session_id: str, held: bool, note: str = "") -> bool:
        """セッションの保全を付ける・外す。記録の無いセッションなら False。"""
        if not session_id or not any(self.session(date.fromisoformat(d), session_id) for d in self.days()):
            return False
        with self._lock:
            holds = self.holds()
            if held:
                holds[session_id] = {"held_at": datetime.now(JST).isoformat(timespec="seconds"), "note": note[:200]}
            else:
                holds.pop(session_id, None)
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / (HOLDS_FILE + ".tmp")
            tmp.write_text(json.dumps(holds, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.root / HOLDS_FILE)
        logger.info("文字記録の保全を%sしました", "設定" if held else "解除")
        return True

    # ---- 期限 ------------------------------------------------------------

    def cleanup(self, today: date | None = None) -> int:
        """保持期間を過ぎた日のファイルを消す（保全したセッションの行だけ残す）。消した行数を返す。"""
        today = today or datetime.now(JST).date()
        keep_from = today - timedelta(days=self.retention_days)
        holds = self.holds()
        removed = 0
        for d in self.days():
            if date.fromisoformat(d) >= keep_from:
                continue
            path = self.root / f"{d}.jsonl"
            rows = self.read(date.fromisoformat(d))
            kept = [r for r in rows if r.get("session_id") in holds]
            removed += len(rows) - len(kept)
            with self._lock:
                if kept:
                    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
                else:
                    path.unlink(missing_ok=True)
        return removed
