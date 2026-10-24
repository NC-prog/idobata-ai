"""用件の聞き分け（WP13 E-3）の返事の判定を、人の見立て（cases_clarify_replies.yaml）と比べる。LLM は使わない。

    cd server && uv run python -m tests.eval.run_clarify [--site etajima]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.clarify import Clarifier, load_clarify_rules
from src.config import SERVER_ROOT, config
from src.procedure_kb import procedures_dir
from src.procedures import load_procedures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="etajima")
    args = ap.parse_args()
    procs = load_procedures(SERVER_ROOT / "context", procedures_dir(config.data_dir, args.site))
    c = Clarifier(procs, load_clarify_rules(SERVER_ROOT / "context" / "actions.yaml"), config.procedure)
    cases = yaml.safe_load((Path(__file__).parent / "cases_clarify_replies.yaml").read_text(encoding="utf-8"))
    ok, held, held_ok = 0, 0, 0
    print("| 問いの発話 | 選択肢 | 返事 | 人の見立て | 判定 | 一致 |")
    print("|---|---|---|---|---|---|")
    for x in cases:
        q = c.find(x["ask"])
        if q is None:
            print(f"| {x['ask']} | （聞き分けに入らない） | {x['reply']} | {x['expected']} | - | × |")
            continue
        got = c.pick(q, x["reply"])
        gid = got if isinstance(got, str) else (got.id if got is not None else "none")
        same = gid == x["expected"]
        ok += same
        if x.get("held_out"):
            held, held_ok = held + 1, held_ok + same
        mark = "（手直しなし）" if x.get("held_out") else ""
        print(f"| {x['ask']} | {' / '.join(c.label(r) for r in q.candidates)} | {x['reply']}{mark} | {x['expected']} | {gid} | "
              f"{'○' if same else '×'} |")
    print(f"\n一致: {ok} / {len(cases)}（{ok * 100 // len(cases)}%）。うち手直しなしの例 {held_ok} / {held}")


if __name__ == "__main__":
    main()
