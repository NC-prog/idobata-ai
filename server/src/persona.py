"""システムプロンプトの組み立て（WP03）。

2 層: 固定事実（拠点 YAML の facility）はそのまま書き、行事・手続きの詳細・天気は
知識パックから引いた【参考資料】【今日】として毎ターン差し込む。
テンプレートは ``config.DialogueConfig.persona_template``。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from .config import DialogueConfig, PersonaConfig
from .knowledge import Reference
from .procedure_kb import PROCEDURE_TRUST
from .site_context import Department, Facility

_WEEKDAYS = "月火水木金土日"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Persona:
    """案内役の人物像（persona.yaml）。"""

    role_title: str
    profile: tuple[str, ...] = ()
    speaking: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()
    # 人物像と食い違う言い方（この語を含む文は読み上げない）
    contradictions: tuple[str, ...] = ()

    def block(self, facility_name: str) -> str:
        """システムプロンプトの【あなたの人物像】の節。"""
        def lines(items: tuple[str, ...]) -> str:
            return "".join(f"- {x.format(role_title=self.role_title, facility_name=facility_name)}\n"
                           for x in items)
        return ("【あなたの人物像】（【話し方の決まり】と【ここで分かっていること】のほうが優先）\n"
                + lines(self.profile) + lines(self.speaking)
                + ("【人物像についての決まり】\n" + lines(self.guardrails) if self.guardrails else ""))


def load_persona(cfg: PersonaConfig) -> Persona | None:
    """persona.yaml を読む。無い・壊れているときは None（人物像なしで動く）。"""
    try:
        data: Any = yaml.safe_load(Path(cfg.path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("人物像の設定を読めません: %s（人物像なしで続けます）", cfg.path, exc_info=True)
        return None

    def strs(key: str) -> tuple[str, ...]:
        # YAML の折り返しで入る空白は、日本語の文字どうしの間では詰める
        return tuple(re.sub(r"(?<=[^\x00-\x7f])\s+(?=[^\x00-\x7f])", "", " ".join(str(x).split()))
                     for x in data.get(key) or [])

    return Persona(cfg.role_title or str(data.get("role_title") or "職員"),
                   strs("profile"), strs("speaking"), strs("guardrails"), strs("contradictions"))


def format_department(d: Department) -> str:
    """「市民課（1階）: 住民票、戸籍」の形。別の建物の課はその場所を書く。"""
    if d.floor:
        where = f"（{d.floor}）"
    elif d.place:
        where = f"（この建物ではなく {d.place}）"
    else:
        where = ""
    handles = "、".join(d.handles)
    return f"{d.name}{where}" + (f": {handles}" if handles else "")


def format_facts(facility: Facility | None) -> str:
    if facility is None:
        return "- （拠点の情報が設定されていません）"
    lines = [f"- 場所: {facility.location_phrase}"]
    if facility.hours:
        lines.append(f"- 開庁時間: {facility.hours}")
    if facility.departments:
        lines.append("- 課と担当:")
        lines += [f"  - {format_department(d)}" for d in facility.departments]
    if facility.notes:
        lines += [f"- {n}" for n in facility.notes]
    lines.append("- 館内の設備（ここに無い設備の場所は分からない）:")
    if facility.facilities_in_building:
        lines += [f"  - {f.name}: {f.location}" for f in facility.facilities_in_building]
    else:
        lines.append("  - （登録なし）")
    if facility.emergency_lines:
        lines.append("- 緊急連絡先（手続きではないので、いつ案内してもよい）:")
        lines += [f"  - {e.name}: {e.number}" for e in facility.emergency_lines]
    return "\n".join(lines)


def format_references(refs: list[Reference], none_text: str, start: int = 1) -> str:
    if not refs:
        return none_text
    return "\n".join(
        f"[{i}] (出典: {r.title}" + (f"／{r.label}" if r.label else "") + f")\n{r.text}"
        for i, r in enumerate(refs, start)
    )


def format_procedures(refs: list[Reference], start: int = 1) -> str:
    """【市の案内（承認済み）】の中身（番号は参考資料と通し）。"""
    return "\n".join(f"[{i}] {r.text}" for i, r in enumerate(refs, start))


def source_labels(cfg: DialogueConfig, facility: Facility | None, refs: list[Reference]) -> str:
    """出典の一言の「〇〇」。使った資料の出どころを重ねずに「や」でつなぐ。"""
    default = cfg.default_source_label.format(org=organization_word(facility))
    labels = list(dict.fromkeys(r.label or default for r in refs if r.trust != PROCEDURE_TRUST))
    return "や".join(labels)


def this_weekend(today: date) -> tuple[date, date]:
    """今週末の土曜・日曜。土曜なら今日と明日、日曜なら昨日と今日。来週末はこの 7 日後。"""
    sat = today + timedelta(days=5 - today.weekday()) if today.weekday() <= 5 else today - timedelta(days=1)
    return sat, sat + timedelta(days=1)


def _weekend_text(label: str, sat: date, sun: date) -> str:
    return f"{label}＝{sat.month}月{sat.day}日（土）・{sun.month}月{sun.day}日（日）"


def format_today(today: date, weather_lines: list[str]) -> str:
    """【今日】の中身。曜日と「今週末」は計算して書く（LLM に数えさせると間違える。WP03 の評価で
    「今週末は10月22日（木曜日）」と言い、1 か月先の行事を「来週」と言った）。"""
    sat, sun = this_weekend(today)
    week = timedelta(days=7)
    lines = [
        f"今日: {today.year}年{today.month}月{today.day}日（{_WEEKDAYS[today.weekday()]}曜日）",
        _weekend_text("今週末", sat, sun),
        _weekend_text("来週末", sat + week, sun + week),
    ]
    lines.append("天気: " + ("".join(weather_lines) if weather_lines else "情報なし"))
    return "\n".join(lines)


def organization_word(facility: Facility | None) -> str:
    """「市のホームページ」「町のホームページ」の 市 / 町。"""
    city = facility.address_city if facility else ""
    return city[-1] if city and city[-1] in "市町村" else "市"


def format_propose_rules(cfg: DialogueConfig, enabled: tuple[str, ...]) -> str:
    rules = [f"  - <propose:{k}>: {cond}\n" for k, cond in cfg.propose_conditions if k in enabled]
    return cfg.propose_rule_head + "".join(rules) if rules else ""


def build_system_prompt(
    cfg: DialogueConfig,
    facility: Facility | None,
    today: date,
    weather_lines: list[str],
    avatar_name: str = "",
    enabled_actions: tuple[str, ...] = (),
    greeting: str = "こんにちは",
    persona: Persona | None = None,
    role_block: str | None = None,
) -> str:
    """システムプロンプト（固定事実 + 決まり + 今日）。参考資料は入れない（build_turn_message）。

    role_block はいまの役（roles.py）の段落で、【話し方の決まり】の後ろに入る。None なら人物像の段落
    （総合受付の役の段落と同じもの）。

    挨拶は時間帯（1 日 3 回だけ変わる）なので、プレフィックスキャッシュはほぼ効いたまま。
    """
    return cfg.persona_template.format(
        facility_name=facility.name if facility else "市役所",
        city=(facility.address_city or facility.name) if facility else "この市",
        foreign_reply=cfg.foreign_reply.rstrip("。"),
        location_phrase=facility.location_phrase if facility else "ロビー",
        facts=format_facts(facility),
        today=format_today(today, weather_lines),
        avatar_name=avatar_name or "（未定）",
        propose_rules=format_propose_rules(cfg, enabled_actions),
        greeting=greeting,
        persona=role_block if role_block is not None else (
            persona.block(facility.name if facility else "市役所") if persona else ""),
    )


def build_turn_message(
    cfg: DialogueConfig, facility: Facility | None, refs: list[Reference], text: str,
    procedure_head: str = "",
) -> str:
    """最新の来訪者メッセージ（参考資料 + 発話）。出典の付け方は参考資料があるときだけ書く。

    手続きナレッジ（承認済み、trust が procedure）の資料は、参考資料より前に procedure_head の見出しで置く（WP09）。
    番号は通し（[出典:1] の印はどちらにも使える）。
    """
    procs = [r for r in refs if r.trust == PROCEDURE_TRUST]
    rest = [r for r in refs if r.trust != PROCEDURE_TRUST]
    head = (procedure_head + format_procedures(procs) + "\n") if procs else ""
    return head + cfg.turn_template.format(
        references=format_references(rest, cfg.no_references_text, start=len(procs) + 1),
        source_rule=cfg.source_rule.format(marker_example=cfg.source_marker_example) if refs else "",
        text=text,
    )
