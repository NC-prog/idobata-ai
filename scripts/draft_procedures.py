"""手続きナレッジの下書きを作る（WP09 B-1）。

    cd server && CUDA_VISIBLE_DEVICES= uv run python ../scripts/draft_procedures.py --site etajima [--only juminhyo ...]

``server/context/procedure_catalog.yaml`` の 50 件について、知識パック（クロール済み）から関連ページを探し、
LLM（standalone では 7560 の Qwen3-8B）に A-1 の形式で下書きさせて ``$DATA/procedures/<site>/<id>.yaml`` に書く。

- 下書きは全部 ``status: draft``。1 件も自動で承認しない（承認は職員パネルで職員がする）
- 根拠の無い項目は null／空。持ち物・流れ・期限・金額・想定問答の答えは、出典のページの本文と文字 2-gram で
  照合し、重なりが足りないもの・本文に無い数や支払方法を含むものは外す（外したものは staff_note に書く）
- 担当課は規則だけで決める（procedure_catalog.yaml の handles を facility の handles と突き合わせる。
  複数ならこの建物の課を優先、それでも決まらなければ空欄にして staff_note に書く）。LLM には課を選ばせない
- ``sources`` に使ったページの URL と確認日、``content_hash`` に出典のページの本文ハッシュ（再確認の検知用）
- 既にあるファイルは書き換えない（承認済み・職員が直したものを守る）。--redraft で下書きだけ作り直す
- procedure_seeds.yaml（WP04 の procedures.yaml から移した 3 件、承認済み）は無ければ書く

最後に要約表（手続き名・担当課・出典の有無・空欄の数）を Markdown で出す（--out）。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from src.config import SERVER_ROOT, KnowledgeConfig, config
from src.dialogue import _is_non_japanese_kanji, grounding_overlap, latest_dated
from src.knowledge import JST, KnowledgeBase
from src.llm_client import LLMClient
from src.persona import organization_word
from src.procedure_kb import (
    ProcedureBook,
    ProcedureRecord,
    content_hash,
    load_seeds,
    procedures_dir,
    record_from_dict,
)
from src.procedures import _department_keywords
from src.site_context import Facility, load_facility

logger = logging.getLogger("draft_procedures")

CATALOG = SERVER_ROOT / "context" / "procedure_catalog.yaml"
SEEDS = SERVER_ROOT / "context" / "procedure_seeds.yaml"
ORG_MARK = "〔市町〕"

# 検索: 下書きの材料は広めに拾い、手続きの語がタイトルか本文にあるページだけ使う
SEARCH_CFG = replace(KnowledgeConfig(), top_k=10, max_chars=20000, vector_candidates=30, bm25_candidates=30,
                     min_vector_score=0.82, min_vector_score_with_keyword=0.76, keyword_rank=10)
MAX_PAGES = 3
PAGE_CHARS = 1800
LLM_MAX_TOKENS = 1400
LLM_TIMEOUT = 180.0
# 照合のしきい値（文字 2-gram の重なり）
MIN_OVERLAP = {"bring": 0.6, "steps": 0.55, "deadline": 0.45, "fee": 0.45, "faq": 0.4, "audience": 0.45}
# 本文に手続きの語がこの回数以上あれば、タイトルに無くても材料にする
MIN_BODY_MENTIONS = 3
CLAIM_RE = re.compile(config.dialogue.grounding_claim_pattern)
REMOTE_TITLE_RE = re.compile(r"郵便|郵送|オンライン|電子申請|LINE|ライン|コンビニ")
# 窓口の持ち物に入れないもの（郵送・オンラインの申し込みだけに要るもの。8B は決まりに書いても入れてくる）
REMOTE_ITEM_RE = re.compile(r"郵便小為替|小為替|返信用封筒|切手|スマートフォン|スマホ|パソコン|クレジット|PayPay|"
                            r"ペイペイ|インターネット|暗証番号|読み取り|アプリ")
_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")

PROMPT = """あなたは役所の窓口の手続きを整理する係です。【資料】（市町の公式ホームページの写し）だけを根拠に、
手続き「{name}」の案内を JSON で下書きしてください。

決まり:
- 【資料】に書いてあることだけを書く。書いていないことは null か空の配列にする。一般的な知識で補わない。
- 持ち物（bring）は、窓口で手続きするときに持って行くものを【資料】の書き方のまま短く（1 項目 30 字以内）。最大 6 項目。
  郵送・オンライン（LINE・マイナポータルなど）で申し込むときだけに要るもの（郵便小為替・返信用封筒・スマートフォンなど）は入れない。
- 流れ（steps）は窓口での流れを 3〜5 行。【資料】に流れが書いていなければ空。
- 資料の表に複数の手続きの金額や期限が並んでいるときは、「{name}」の行のものだけを使う。
- 期限（deadline）・金額や手数料（fee）は、【資料】に数字つきで明記があるときだけ、その言い方のまま。無ければ null。
- 対象（audience）はどんな人の手続きかを 1 行で（【資料】に無ければ null）。
- 想定問答（faq）は 2〜4 組。来訪者が窓口で聞きそうな話し言葉の問いと、【資料】に沿った 1〜2 文の答え。
  答えに【資料】に無い期限・金額・持ち物を書かない。
- 紹介状の聞き取り（intake）は、担当の職員が準備できるように来訪者に伺うこと 0〜3 問（短い問いの形）。
  氏名・住所・電話番号・マイナンバーは聞かない。
- used は根拠に使った資料の番号。どれも関係なければ空の配列にし、ほかの項目も空にする。
出力は JSON だけ:
{{"audience": null, "bring": [], "steps": [], "deadline": null, "fee": null,
 "faq": [{{"q": "", "a": ""}}], "intake": [{{"key": "", "question": ""}}], "used": [1]}}
"""


def load_catalog(site: str, facility: Facility | None) -> list[dict[str, Any]]:
    org = organization_word(facility)
    text = CATALOG.read_text(encoding="utf-8").replace(ORG_MARK, org)
    return list((yaml.safe_load(text) or {}).get("catalog") or [])


def pages_of_pack(pack_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """パックの source_url → チャンク（順に）と、source_url → ページの本文ハッシュ。"""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    pages: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for r in pq.read_table(pack_dir / "chunks.parquet").to_pylist():
        pages.setdefault(r["source_url"], []).append(r)
        hashes[r["source_url"]] = str(r.get("page_hash") or "")
    return pages, hashes


def relevant(entry: dict[str, Any], title: str, text: str) -> bool:
    """ページが手続きの材料になるか（タイトルに手続きの語があるか、本文に MIN_BODY_MENTIONS 回以上あるか）。

    本文に一度出てくるだけのページ（情報公開請求のページの「戸籍」など）は使わない。
    """
    words = topic_words(entry)
    if any(w in title for w in words):
        return True
    return sum(text.count(w) for w in words) >= MIN_BODY_MENTIONS


def catalog_department(entry: dict[str, Any], facility: Facility | None) -> tuple[str | None, str]:
    """担当課（規則だけ）と、決まらなかったときの理由。"""
    if facility is None:
        return None, "拠点の課の情報がありません"
    hints = set(entry.get("handles") or [])
    names = set(entry.get("dept_names") or [])
    cands = [d for d in facility.departments if d.name in names or hints & _department_keywords(d.handles)]
    if len(cands) > 1:
        here = [d for d in cands if not d.place]
        cands = here if len(here) == 1 else cands
    if len(cands) == 1:
        return cands[0].name, ""
    if not cands:
        return None, "担当課は規則で決まりませんでした（拠点の課の handles に当たる語がありません。職員が付けてください）"
    return None, ("担当課は規則で一意に決まりませんでした（候補: " + "、".join(d.name for d in cands)
                  + "。職員が付けてください）")


def find_pages(kb: KnowledgeBase, entry: dict[str, Any], pages: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    """手続きの材料になるページ（最大 MAX_PAGES、本文は PAGE_CHARS まで）。"""
    q = f"{entry['name']} {entry.get('query') or ''}"
    res = kb.search(q)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for ref in res.references:
        url = ref.source_url
        if url in seen or url not in pages:
            continue
        seen.add(url)
        chunks = pages[url]
        text = "\n".join(c["text"] for c in chunks)
        if not relevant(entry, ref.title, text):
            continue
        out.append({"url": url, "title": ref.title, "text": text[:PAGE_CHARS], "full": text})
    # タイトルに手続きの語があるページを先に（「住民票」の問いに郵便請求のページより窓口のページを）
    words = [w for w in [re.sub(r"[（(].*?[）)]", "", entry["name"]), *(entry.get("aliases") or [])] if len(w) >= 2]
    # 郵送・オンラインの申し込みのページは後ろに（窓口の案内を下書きの材料の先頭にする）
    out.sort(key=lambda p: (not any(w in p["title"] for w in words), bool(REMOTE_TITLE_RE.search(p["title"]))))
    return out[:MAX_PAGES]


def parse_json(raw: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def topic_words(entry: dict[str, Any]) -> list[str]:
    """手続きの語（名前を「・」や括弧で分けたものと別名。2 文字以上）。"""
    name = re.sub(r"[（(].*?[）)]", "", entry["name"])
    parts = [name, *re.split(r"[・、]", name), *(entry.get("aliases") or [])]
    # 「戸籍謄本・抄本の交付」→ 戸籍謄本、抄本、…。語尾の「の交付」「の申請」などは外す
    parts = [re.sub(r"の(交付|申請|届出|相談|納付|入居|加入|登録)$", "", p) for p in parts]
    return [p for p in dict.fromkeys(parts) if len(p) >= 2]


def grounded(value: str, sources: list[str], kind: str, words: list[str] | None = None) -> bool:
    """下書きの 1 項目が、出典の本文に根拠を持つか（重なりと、数・支払方法の語がそのままあるか）。

    期限・金額は、数が手続きの語と同じ行にあることも要る（表の別の行の金額を拾わないように。
    戸籍謄本の手数料に、同じ表の住民票の 200円 を書いた）。
    """
    v = value.translate(_ZEN)
    srcs = [s.translate(_ZEN) for s in sources]
    best = max((grounding_overlap(v, s) for s in srcs), default=0.0)
    claims = CLAIM_RE.findall(v)
    missing = [c for c in claims if not any(c in s for s in srcs)]
    if best < MIN_OVERLAP[kind] or missing:
        return False
    if kind in ("deadline", "fee") and words and claims:
        rows = [ln for s in srcs for ln in s.splitlines()]
        return all(any(c in ln and any(w in ln for w in words) for ln in rows) for c in claims)
    return True


def stale(text: str, today: str) -> bool:
    """書いてある年付きの日付がすべて今日より前か（去年の募集期間・締切を下書きに入れない）。"""
    d = latest_dated(text)
    return d is not None and d.isoformat() < today


def lint(site: str, today: str) -> list[str]:
    """既にある下書きを LLM なしで点検し直す（過ぎた日付・中国語の漢字の項目を外す）。直した id を返す。"""
    book = ProcedureBook(procedures_dir(config.data_dir, site, config.procedure.subdir))
    fixed = []
    for r in book.records():
        if r.status != "draft":
            continue
        bad: list[str] = []

        def ok(v: str, bad: list[str] = bad) -> bool:
            if stale(v, today) or any(_is_non_japanese_kanji(ch) for ch in v):
                bad.append(v)
                return False
            return True

        new = replace(
            r, bring=tuple(b for b in r.bring if ok(b.item)), steps=tuple(x for x in r.steps if ok(x)),
            faq=tuple(f for f in r.faq if ok(f[1])), deadline=r.deadline if r.deadline and ok(r.deadline) else None,
            fee=r.fee if r.fee and ok(r.fee) else None, audience=r.audience if ok(r.audience) else "")
        if not bad:
            continue
        note = "下書きの点検で、過ぎた日付か中国語の漢字を含むので外した項目: " + "／".join(bad)
        book.save(replace(new, staff_note="\n".join(x for x in (r.staff_note, note) if x)))
        fixed.append(r.id)
    return fixed


def draft_one(llm: LLMClient, entry: dict[str, Any], facility: Facility | None,
              found: list[dict[str, str]], hashes: dict[str, str], today: str) -> tuple[ProcedureRecord, dict[str, Any]]:
    """1 件の下書き（record, 点検の記録）。"""
    note: list[str] = []
    stats: dict[str, Any] = {"dropped": [], "llm_failed": False}
    dept, why = catalog_department(entry, facility)
    if why:
        note.append(why)
    data: dict[str, Any] = {}
    if found:
        material = "\n\n".join(f"[{i}] {p['title']}（{p['url']}）\n{p['text']}" for i, p in enumerate(found, 1))
        prompt = PROMPT.format(name=entry["name"])
        for attempt in range(2):
            try:
                raw = llm.chat(config.llm.chat_model,
                               [{"role": "system", "content": prompt},
                                {"role": "user", "content": f"【資料】\n{material}\n\n手続き: {entry['name']}"}],
                               max_tokens=LLM_MAX_TOKENS, temperature=0.0, timeout=LLM_TIMEOUT,
                               extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            except Exception:
                logger.warning("LLM に失敗しました（%s、%d 回目）", entry["id"], attempt + 1, exc_info=True)
                continue
            parsed = parse_json(raw)
            if parsed is not None:
                data = parsed
                break
        else:
            stats["llm_failed"] = True
            note.append("下書きの自動生成に失敗しました（LLM の出力を読めませんでした）")
    used_idx = [int(i) for i in data.get("used") or [] if str(i).isdigit() and 1 <= int(i) <= len(found)]
    used = [found[i - 1] for i in dict.fromkeys(used_idx)] or (found if data else [])
    src_text = [p["full"] for p in used]
    words = topic_words(entry)

    def keep(kind: str, value: Any) -> str | None:
        s = str(value).strip() if value is not None else ""
        if not s or s.lower() in ("null", "none", "なし"):
            return None
        if stale(s, today):
            stats["dropped"].append((kind, s))
            return None
        if any(_is_non_japanese_kanji(ch) for ch in s):
            stats["dropped"].append((kind, s))  # 中国語の漢字（簡体字）が混じった（「夫妇」）
            return None
        if grounded(s, src_text, kind, words):
            return s
        stats["dropped"].append((kind, s))
        return None

    remote = [str(b) for b in data.get("bring") or [] if REMOTE_ITEM_RE.search(str(b))]
    if remote:
        note.append("郵送・オンラインの申し込みだけに要るものとして持ち物から外した: " + "／".join(remote))
    bring = [x for x in (keep("bring", b) for b in (data.get("bring") or [])
                         if not REMOTE_ITEM_RE.search(str(b))) if x][:6]
    steps = [x for x in (keep("steps", s) for s in (data.get("steps") or [])[:5]) if x]
    faq = []
    for f in (data.get("faq") or [])[:4]:
        if isinstance(f, dict) and f.get("q") and keep("faq", f.get("a")):
            faq.append({"q": str(f["q"]).strip(), "a": str(f["a"]).strip()})
    intake = [{"key": str(q.get("key") or f"q{i + 1}"), "question": str(q["question"]).strip()}
              for i, q in enumerate((data.get("intake") or [])[:3])
              if isinstance(q, dict) and q.get("question") and len(str(q["question"])) <= 40
              and not re.search(r"氏名|名前|住所|電話|マイナンバー|番号", str(q["question"]))]
    audience = keep("audience", data.get("audience")) or ""
    deadline, fee = keep("deadline", data.get("deadline")), keep("fee", data.get("fee"))
    if stats["dropped"]:
        note.append("下書きの自動点検で、出典の本文に根拠が見つからず外した項目: "
                    + "／".join(f"{k}: {v}" for k, v in stats["dropped"]))
    if not found:
        note.append("知識パックに該当するページが見つかりませんでした（出典なし）")
    rec = record_from_dict({
        "id": entry["id"], "name": entry["name"], "aliases": entry.get("aliases") or [],
        "when": entry.get("when") or [], "tags": entry.get("tags") or [], "department": dept or "",
        "audience": audience, "bring": bring, "steps": steps, "deadline": deadline,
        "fee": fee, "related": [], "intake": intake, "faq": faq,
        "sources": [{"url": p["url"], "checked_at": today} for p in used],
        "status": "draft", "staff_note": "\n".join(note) or None,
        "content_hash": content_hash(hashes.get(p["url"], "") for p in used) if used else None,
    })
    stats["pages"] = [p["url"] for p in found]
    return rec, stats


def run(site: str, only: list[str], redraft: bool, limit: int | None) -> list[dict[str, Any]]:
    facility = load_facility(site, config.asr.site_context_dir)
    book = ProcedureBook(procedures_dir(config.data_dir, site, config.procedure.subdir))
    for seed in load_seeds(SEEDS, site):
        if book.get(seed.id) is None:
            book.save(seed)
            logger.info("移行した手続きを書きました（承認済み）: %s", seed.id)
    pack_dir = (config.data_dir / "kb" / site).resolve()
    kb = KnowledgeBase(pack_dir, SEARCH_CFG)
    pages, hashes = pages_of_pack(pack_dir)
    llm = LLMClient()
    today = datetime.now(JST).date().isoformat()
    report: list[dict[str, Any]] = []
    entries = [e for e in load_catalog(site, facility) if not only or e["id"] in only]
    for n, entry in enumerate(entries):
        if limit is not None and n >= limit:
            break
        current = book.get(entry["id"])
        if current is not None and (current.status != "draft" or not redraft):
            report.append({"record": current, "stats": {"kept": True, "dropped": [], "pages": []}})
            continue
        t0 = time.monotonic()
        found = find_pages(kb, entry, pages)
        rec, stats = draft_one(llm, entry, facility, found, hashes, today)
        book.save(rec)
        stats["sec"] = time.monotonic() - t0
        logger.info("下書き %s: 出典 %d、外した項目 %d、%.1f 秒", entry["id"], len(rec.sources),
                    len(stats["dropped"]), stats["sec"])
        report.append({"record": rec, "stats": stats})
    return report


def summary_table(site: str, report: list[dict[str, Any]]) -> str:
    rows = [f"### {site}", "",
            "| id | 手続き名 | 担当課 | 状態 | 出典 | 持ち物 | 流れ | 期限 | 金額 | 問答 | 空欄 | 外した項目 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    no_source = 0
    dropped = 0
    for item in report:
        r: ProcedureRecord = item["record"]
        st = item["stats"]
        if not r.sources:
            no_source += 1
        dropped += len(st.get("dropped") or [])
        rows.append(
            f"| {r.id} | {r.name} | {r.department or '（未定）'} | {r.status} | "
            f"{len(r.sources) or 'なし'} | {len(r.bring)} | {len(r.steps)} | {'有' if r.deadline else '-'} | "
            f"{'有' if r.fee else '-'} | {len(r.faq)} | {len(r.blank_fields())} | {len(st.get('dropped') or [])} |")
    undecided = sum(1 for i in report if not i["record"].department)
    rows += ["", f"出典なし {no_source} 件 / {len(report)} 件。担当課が未定 {undecided} 件。自動点検で外した項目 {dropped} 個。"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", action="append", help="拠点（複数可。省略で .env の IDOBATA_SITE）")
    ap.add_argument("--only", nargs="*", default=[], help="この id だけ")
    ap.add_argument("--redraft", action="store_true", help="下書き（draft）のものは作り直す（承認済み・要再確認は触らない）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="要約表を書き出す Markdown")
    ap.add_argument("--report-json", type=Path, default=None, help="点検の記録（外した項目など）を JSON で")
    ap.add_argument("--lint", action="store_true", help="LLM を使わず、既にある下書きを点検し直すだけ")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.lint:
        today = datetime.now(JST).date().isoformat()
        for site in args.site or [config.site]:
            print(site, "点検で直した下書き:", " ".join(lint(site, today)) or "なし")
        return
    tables = []
    dump: dict[str, Any] = {}
    for site in args.site or [config.site]:
        report = run(site, args.only, args.redraft, args.limit)
        tables.append(summary_table(site, report))
        dump[site] = [{"id": i["record"].id, "dropped": i["stats"].get("dropped"), "pages": i["stats"].get("pages"),
                       "kept": i["stats"].get("kept", False)}
                      for i in report]
    text = "\n\n".join(tables)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.report_json:
        args.report_json.write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
