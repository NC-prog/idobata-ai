"""WP03 評価セットを LLM に掛けて、結果を Markdown の表で出す（音声は使わない）。

    cd server && uv run python -m tests.eval.run [--site etajima] [--out 結果.md]

実サービス（LLM）と知識パックを使う。1 問ごとに履歴を消して、独立した会話として聞く。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

import yaml

from src.config import config
from src.dialogue import Dialogue
from src.knowledge import KnowledgeBase, Weather
from src.llm_client import LLMClient
from src.site_context import load_facility

_ASCII_RUN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]{3,}")


def _auto_check(reply: str, auto: dict[str, Any], allow: set[str]) -> list[str]:
    notes: list[str] = []
    must_any = auto.get("must_any") or []
    if must_any and not any(k in reply for k in must_any):
        notes.append(f"必須語なし（{'/'.join(must_any)}）")
    for k in auto.get("must_not") or []:
        if k in reply:
            notes.append(f"禁止語「{k}」")
    ascii_words = [w for w in _ASCII_RUN.findall(reply) if w.lower() not in allow]
    if ascii_words:
        notes.append(f"英字 {ascii_words}")
    return notes


async def run(site: str) -> str:
    facility = load_facility(site, config.asr.site_context_dir)
    kb = KnowledgeBase.load(config.data_dir, site, config.knowledge)
    dialogue = Dialogue(LLMClient(), facility=facility, knowledge=kb,
                        weather=Weather.for_site(config.data_dir, site))
    allow = {a.lower() for a in (facility.allow_ascii if facility else ())}
    cases = yaml.safe_load(
        (Path(__file__).parent / f"cases_{site}.yaml").read_text(encoding="utf-8")
    )
    rows = ["| # | 発話 | 応答（全文） | 参考資料（上位） | 出典表示 | 検索 | 補助判定 |",
            "|---|---|---|---|---|---|---|"]
    ascii_total = 0
    for case in cases:
        dialogue.reset()
        t0 = time.monotonic()
        reply = "".join(
            [s async for s in dialogue.respond(case["utterance"], case.get("language", "ja"))]
        )
        elapsed = time.monotonic() - t0
        notes = _auto_check(reply, case.get("auto") or {}, allow)
        ascii_total += sum(1 for n in notes if n.startswith("英字"))
        refs = " / ".join(f"{r.title}（{r.vector_score:.2f}）" for r in dialogue.last_references[:3])
        cited = " / ".join(r.title for r in dialogue.last_cited)
        search = "-" if dialogue.last_search_ms is None else f"{dialogue.last_search_ms:.0f}ms"
        rows.append(
            f"| {case['id']} | {case['utterance']} | {reply} | {refs or 'なし'} | {cited or '-'} | "
            f"{search} | {'OK' if not notes else '; '.join(notes)}（{elapsed:.1f}s） |"
        )
        print(f"[{case['id']}] {case['utterance']}\n  → {reply}\n  参考: {refs or 'なし'}  補助: "
              f"{notes or 'OK'}", flush=True)
    rows.append("")
    rows.append(f"英字の混在: {ascii_total} 問")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=config.site)
    ap.add_argument("--out", type=Path, default=None, help="表を書き出す Markdown ファイル")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    table = asyncio.run(run(args.site))
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"書き出し: {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
