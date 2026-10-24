"""10/24 の実機の 4 つの脱線を会話の流れごと流し、渡した資料と応答に脱線の語が出ないかを見る（WP13 C）。

    cd server && uv run python -m tests.eval.run_regression [--site akiota]

実サービス（LLM）と知識パックを使う。1 件ごとに会話を区切る。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from tests.eval.run_wp09 import _build


async def run(site: str) -> int:
    cases = yaml.safe_load((Path(__file__).parent / "cases_regression_1024.yaml").read_text(encoding="utf-8"))
    conv, dialogue, _facility, _procs = _build(site)
    bad = 0
    for case in cases:
        if case.get("site", site) != site:
            continue
        await conv.end_session()
        hits: list[str] = []
        for text in case["turns"]:
            reply = "".join([s async for s in conv.respond(text, "ja")])
            titles = [r.title for r in dialogue.last_references]
            found = [w for w in case["off_topic"] if w in reply or any(w in t for t in titles)]
            hits += found
            print(f"[{case['id']}] {text}\n  → {reply}\n  型: {getattr(dialogue, 'last_intent', '-')}  資料: {titles}"
                  f"  脱線の語: {found or 'なし'}", flush=True)
        bad += bool(hits)
        print(f"== {case['id']}: {'脱線あり ' + str(sorted(set(hits))) if hits else '脱線なし'}\n", flush=True)
    print(f"脱線した流れ: {bad} / {len(cases)}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="akiota")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args.site))


if __name__ == "__main__":
    main()
