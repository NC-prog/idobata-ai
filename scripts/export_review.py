"""会話の文字記録から、誤り探しの素材を抜き出す（WP06）。

    uv run --project server python scripts/export_review.py [--site etajima] [--date 2026-10-22]

``$DATA/transcripts/<site>/<date>.jsonl`` を読み、次の 4 種類を ``$DATA/review/<date>.jsonl`` に 1 件 1 行で書く
（同じ日をもう一度流すと上書き）。将来 T2 で評価・学習に使う。

- asr_fixed: 聞き取り間違いの対応表で直った発話（ASR の生テキストと直した後）
- dropped: 安全装置で読み上げなかった文（理由つき）
- declined: 提案が辞退された往復（提案した往復と、辞退の返事）
- rephrased: 来訪者が言い直した往復（直前の来訪者の発話と文字 2-gram の重なりが高い）

氏名・電話番号・地区（拠点 YAML の地名と番地）は「＊＊」に伏せる。文字記録の時点で、紹介状の聞き取りで
伺った答えは「（お名前を伺いました）」に置き換わっているが、ふだんの会話の中の名乗りや番号はここで伏せる。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTEXT_DIR = REPO_ROOT / "server" / "context"
JST = timezone(timedelta(hours=9))

# 言い直しとみなす、直前の発話との文字 2-gram の重なり（今の発話の 2-gram のうち直前にもある割合）と最小の長さ
REPHRASE_MIN_OVERLAP = 0.5
REPHRASE_MIN_CHARS = 5
# 伏せた跡
MASK = "＊＊"

_PHONE_RE = re.compile(r"[0-9０-９〇零一二三四五六七八九ゼロマル][0-9０-９〇零一二三四五六七八九ゼロマル\-ー－‐の、\s]{5,}"
                       r"[0-9０-９〇零一二三四五六七八九]")
_NAME_RES = (
    re.compile(r"((?:私|わたし|わたくし|僕|ぼく|俺|おれ|わし|名前)(?:は|が)?、?)[一-龥ぁ-んァ-ヶ]{1,6}"
               r"(?=です|と申します|といいます|と言います|って言います|でございます)"),
    re.compile(r"[一-龥]{1,4}(?=さん|様|さま|くん|ちゃん)"),
)
_ADDRESS_RE = re.compile(r"[0-9０-９一二三四五六七八九十〇百千]+\s*(?:丁目|番地|番|号)")


def place_names(site: str) -> list[str]:
    """伏せる地名（拠点 YAML の place_names と offices の「〜町」。市の名前そのものは伏せない）。"""
    try:
        data: Any = yaml.safe_load((CONTEXT_DIR / f"{site}.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    site_name = str(data.get("site_name") or "")
    words: list[str] = []
    for item in data.get("place_names") or []:
        surface = item.get("surface") if isinstance(item, dict) else item
        if isinstance(surface, str) and surface and surface not in site_name:
            words.append(surface)
    words += [str(o) for o in data.get("offices") or [] if isinstance(o, str) and o.endswith("町")]
    return sorted(set(words), key=len, reverse=True)


def redact(text: str, places: list[str]) -> str:
    out = _PHONE_RE.sub(MASK, text or "")
    for pat in _NAME_RES:
        out = pat.sub(lambda m: (m.group(1) if m.groups() else "") + MASK, out)
    out = _ADDRESS_RE.sub(MASK, out)
    for p in places:
        out = out.replace(p, MASK)
    return out


def bigrams(text: str) -> set[str]:
    t = re.sub(r"[\s、。，．,.！？!?「」…ー]", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def overlap(previous: str, current: str) -> float:
    cur = bigrams(current)
    return len(cur & bigrams(previous)) / len(cur) if cur else 0.0


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def extract(rows: list[dict[str, Any]], places: list[str]) -> list[dict[str, Any]]:
    """1 日分の文字記録から誤り探しの素材を抜き出す（セッションごと、時刻順）。"""
    def r(text: Any) -> str:
        return redact(str(text or ""), places)

    def turn(u: dict[str, Any] | None, a: dict[str, Any] | None) -> dict[str, Any]:
        return {"user": r(u.get("text")) if u else None, "assistant": r(a.get("text")) if a else None}

    out: list[dict[str, Any]] = []
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sessions.setdefault(str(row.get("session_id") or ""), []).append(row)
    for sid, items in sessions.items():
        items.sort(key=lambda x: str(x.get("ts")))
        base = {"session_id": sid}
        last_user: dict[str, Any] | None = None
        last_assistant: dict[str, Any] | None = None
        proposing: tuple[dict[str, Any] | None, dict[str, Any]] | None = None
        for row in items:
            speaker = row.get("speaker")
            if speaker == "user":
                if row.get("ignored"):
                    continue
                text = str(row.get("text") or "")
                if row.get("asr_raw") and row.get("asr_raw") != text:
                    out.append({**base, "kind": "asr_fixed", "ts": row.get("ts"),
                                "asr_raw": r(row["asr_raw"]), "text": r(text)})
                if (last_user is not None and len(re.sub(r"\s", "", text)) >= REPHRASE_MIN_CHARS
                        and overlap(str(last_user.get("text") or ""), text) >= REPHRASE_MIN_OVERLAP):
                    out.append({**base, "kind": "rephrased", "ts": row.get("ts"),
                                "previous": turn(last_user, last_assistant), "rephrased": r(text),
                                "overlap": round(overlap(str(last_user.get("text") or ""), text), 2)})
                last_user, last_assistant = row, None
            elif speaker == "assistant":
                last_assistant = row
                for d in row.get("dropped") or []:
                    out.append({**base, "kind": "dropped", "ts": row.get("ts"), "reason": d.get("reason"),
                                "text": r(d.get("text")), "context": turn(last_user, None)})
                info = row.get("proposal") or {}
                if info.get("proposed"):
                    proposing = (last_user, row)
                if info.get("outcome") == "declined" and proposing is not None:
                    out.append({**base, "kind": "declined", "ts": row.get("ts"), "action": info.get("reply_to"),
                                "proposed_turn": turn(*proposing), "reply": turn(last_user, row)})
                    proposing = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--site", default=os.environ.get("IDOBATA_SITE", "etajima"))
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(JST).date())
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("IDOBATA_DATA_DIR", str(Path.home() / "idobata-data"))))
    args = parser.parse_args()

    src = args.data_dir / "transcripts" / args.site / f"{args.date.isoformat()}.jsonl"
    if not src.is_file():
        print(f"文字記録がありません: {src}", file=sys.stderr)
        return 1
    items = extract(read_rows(src), place_names(args.site))
    dst = args.data_dir / "review" / f"{args.date.isoformat()}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps({"date": args.date.isoformat(), "site": args.site, **x}, ensure_ascii=False)
                           + "\n" for x in items), encoding="utf-8")
    counts: dict[str, int] = {}
    for x in items:
        counts[x["kind"]] = counts.get(x["kind"], 0) + 1
    print(f"{dst}: {len(items)} 件 {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
