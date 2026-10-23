"""手続きナレッジの下書きを開発者がまとめて仮に承認する（人間が明示的に実行するスクリプト。対話の経路からは呼ばない）。

    cd server && uv run python ../scripts/approve_procedures.py --site etajima --provisional          # 一覧だけ
    cd server && uv run python ../scripts/approve_procedures.py --site etajima --provisional --yes    # 実行

対象は、状態が下書きで、出典（sources）が 1 件以上あり、持ち物（bring）か手順（steps）のどちらかが埋まっているもの。
approved_by を ``developer（仮承認）``、approved_at を今の時刻にする。出典なし・中身が空のものは対象にしない。
職員パネルでは「仮」のバッジが付き、職員が「承認」を押し直すとログイン名の通常の承認に置き換わる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from src.config import config
from src.procedure_kb import PROVISIONAL_BY, ProcedureBook, procedures_dir


def run(site: str, yes: bool, directory: Path | None = None) -> tuple[int, int]:
    """(仮承認した・する件数, 除外した件数) を返す。directory は既定で ~/idobata-data/procedures/<site>。"""
    book = ProcedureBook(directory or procedures_dir(config.data_dir, site, config.procedure.subdir))
    targets, skipped = [], []
    for r in sorted(book.records(), key=lambda r: r.id):
        reason = r.provisional_skip_reason()
        if reason is None:
            targets.append(r)
        else:
            skipped.append((r, reason))
    print(f"== {site}: 仮承認の対象 {len(targets)} 件 ==")
    for r in targets:
        print(f"  {r.id:<20} {r.name}（出典 {len(r.sources)}・持ち物 {len(r.bring)}・手順 {len(r.steps)}）")
    print(f"== {site}: 対象外 {len(skipped)} 件 ==")
    for r, reason in skipped:
        print(f"  {r.id:<20} {r.name}: {reason}")
    if not yes:
        print("（--yes が無いので何もしていません）")
        return len(targets), len(skipped)
    for r in targets:
        book.approve(r.id, PROVISIONAL_BY)
    print(f"{site}: {len(targets)} 件を「{PROVISIONAL_BY}」で承認しました")
    return len(targets), len(skipped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", action="append", required=True, choices=["etajima", "akiota"])
    ap.add_argument("--provisional", action="store_true", help="開発者の仮承認として承認する（必須）")
    ap.add_argument("--yes", action="store_true", help="一覧を見たうえで実行する")
    args = ap.parse_args()
    if not args.provisional:
        print("--provisional を付けてください（このスクリプトは開発者の仮承認だけを行います）", file=sys.stderr)
        return 2
    for site in args.site:
        run(site, args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
