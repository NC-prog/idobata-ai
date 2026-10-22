"""WP03 評価セットを LLM に掛けて、結果を Markdown の表で出す（音声は使わない）。

    cd server && uv run python -m tests.eval.run [--site etajima] [--set smalltalk] [--out 結果.md]

実サービス（LLM）と知識パックを使う。1 問ごとに会話を区切って、独立した会話として聞く。
WP06 から実機と同じく Conversation を通す（行為の提案文と、規則による提案の補正が入る）。auto の propose で
提案した行為を確かめる。紹介状などは一時ディレクトリに書く（提案だけで、保存までは進まない）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from src.actions import Conversation
from src.config import SERVER_ROOT, config
from src.dialogue import Dialogue, ends_with_closing
from src.knowledge import KnowledgeBase, Weather
from src.llm_client import LLMClient
from src.persona import load_persona
from src.procedures import load_procedures
from src.referrals import ReferralStore, load_tags
from src.site_context import load_facility

_ASCII_RUN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]{3,}")


def _auto_check(reply: str, auto: dict[str, Any], allow: set[str], proposed: str | None = None) -> list[str]:
    notes: list[str] = []
    must_any = auto.get("must_any") or []
    if must_any and not any(k in reply for k in must_any):
        notes.append(f"必須語なし（{'/'.join(must_any)}）")
    for k in auto.get("must_not") or []:
        if k in reply:
            notes.append(f"禁止語「{k}」")
    ascii_words = [w for w in _ASCII_RUN.findall(reply) if w.lower() not in allow]
    if auto.get("propose") and auto["propose"] != proposed:
        notes.append(f"提案なし（{auto['propose']} を期待、実際は {proposed or 'なし'}）")
    if ascii_words:
        notes.append(f"英字 {ascii_words}")
    return notes


async def run(site: str, case_set: str = "") -> str:
    facility = load_facility(site, config.asr.site_context_dir)
    kb = KnowledgeBase.load(config.data_dir, site, config.knowledge)
    # 実機と同じ条件（呼び名・行為のタグの決まり・人物像）で聞く
    llm = LLMClient()
    dialogue = Dialogue(llm, facility=facility, knowledge=kb,
                        weather=Weather.for_site(config.data_dir, site),
                        avatar_name=config.actions.avatar_name, enabled_actions=config.actions.enabled,
                        persona=load_persona(config.persona),
                        procedures=load_procedures(config.asr.site_context_dir))
    allow = {a.lower() for a in (facility.allow_ascii if facility else ())}
    cases = yaml.safe_load(
        (Path(__file__).parent / f"cases_{site}{'_' + case_set if case_set else ''}.yaml").read_text(encoding="utf-8")
    )
    conv = Conversation(dialogue, llm, config.llm.chat_model, config.actions, config.referral, facility,
                        load_tags(SERVER_ROOT / "context" / "tags.yaml"),
                        referrals=ReferralStore(Path(tempfile.mkdtemp(prefix="idobata-eval-")), site, 1, False),
                        procedures=load_procedures(config.asr.site_context_dir))
    rows = ["| # | 発話 | 応答（全文） | 参考資料（上位） | 出典表示 | 検索 | 提案 | 補助判定 |",
            "|---|---|---|---|---|---|---|---|"]
    ascii_total = 0
    closing_total = 0
    for case in cases:
        await conv.end_session()
        t0 = time.monotonic()
        reply = "".join(
            [s async for s in conv.respond(case["utterance"], case.get("language", "ja"))]
        )
        elapsed = time.monotonic() - t0
        proposed = conv.turn_info.get("proposed")
        notes = _auto_check(reply, case.get("auto") or {}, allow, proposed)
        ascii_total += sum(1 for n in notes if n.startswith("英字"))
        if ends_with_closing(reply, config.dialogue):
            closing_total += 1
            notes.append("締めの定型で終わる")
        refs = " / ".join(f"{r.title}（{r.vector_score:.2f}）" for r in dialogue.last_references[:3])
        cited = " / ".join(r.title for r in dialogue.last_cited)
        search = "-" if dialogue.last_search_ms is None else f"{dialogue.last_search_ms:.0f}ms"
        rows.append(
            f"| {case['id']} | {case['utterance']} | {reply} | {refs or 'なし'} | {cited or '-'} | "
            f"{search} | {proposed or '-'} | {'OK' if not notes else '; '.join(notes)}（{elapsed:.1f}s） |"
        )
        print(f"[{case['id']}] {case['utterance']}\n  → {reply}\n  参考: {refs or 'なし'}  提案: {proposed or '-'}  補助: "
              f"{notes or 'OK'}", flush=True)
    rows.append("")
    rows.append(f"英字の混在: {ascii_total} 問")
    rows.append(f"締めの定型で終わる応答: {closing_total} / {len(cases)} 問")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=config.site)
    ap.add_argument("--set", default="", help="評価セットの種類（smalltalk = 雑談。省略で WP03 の 20 問）")
    ap.add_argument("--out", type=Path, default=None, help="表を書き出す Markdown ファイル")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    table = asyncio.run(run(args.site, args.set))
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"書き出し: {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
