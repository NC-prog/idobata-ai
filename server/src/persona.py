"""システムプロンプトの組み立て（WP03）。

2 層: 固定事実（拠点 YAML の facility）はそのまま書き、行事・手続きの詳細・天気は
知識パックから引いた【参考資料】【今日】として毎ターン差し込む。
テンプレートは ``config.DialogueConfig.persona_template``。
"""

from __future__ import annotations

from datetime import date

from .config import DialogueConfig
from .knowledge import Reference
from .site_context import Department, Facility

_WEEKDAYS = "月火水木金土日"


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
    if facility.emergency_lines:
        lines.append("- 緊急連絡先（手続きではないので、いつ案内してもよい）:")
        lines += [f"  - {e.name}: {e.number}" for e in facility.emergency_lines]
    return "\n".join(lines)


def format_references(refs: list[Reference], none_text: str) -> str:
    if not refs:
        return none_text
    return "\n".join(f"[{i}] (出典: {r.title})\n{r.text}" for i, r in enumerate(refs, 1))


def format_today(today: date, weather_lines: list[str]) -> str:
    head = f"{today.year}年{today.month}月{today.day}日（{_WEEKDAYS[today.weekday()]}曜日）"
    if not weather_lines:
        return f"{head}\n天気: 情報なし"
    return head + "\n天気: " + "".join(weather_lines)


def organization_word(facility: Facility | None) -> str:
    """「市のホームページ」「町のホームページ」の 市 / 町。"""
    city = facility.address_city if facility else ""
    return city[-1] if city and city[-1] in "市町村" else "市"


def build_system_prompt(
    cfg: DialogueConfig,
    facility: Facility | None,
    today: date,
    weather_lines: list[str],
) -> str:
    """システムプロンプト（固定事実 + 決まり + 今日）。参考資料は入れない（build_turn_message）。"""
    return cfg.persona_template.format(
        facility_name=facility.name if facility else "市役所",
        city=(facility.address_city or facility.name) if facility else "この市",
        foreign_reply=cfg.foreign_reply.rstrip("。"),
        location_phrase=facility.location_phrase if facility else "ロビー",
        facts=format_facts(facility),
        today=format_today(today, weather_lines),
    )


def build_turn_message(
    cfg: DialogueConfig, facility: Facility | None, refs: list[Reference], text: str
) -> str:
    """最新の来訪者メッセージ（参考資料 + 発話）。出典の付け方は参考資料があるときだけ書く。"""
    return cfg.turn_template.format(
        references=format_references(refs, cfg.no_references_text),
        source_rule=cfg.source_rule.format(
            source_phrase=cfg.source_phrase.format(org=organization_word(facility)),
            marker_example=cfg.source_marker_example,
        ) if refs else "",
        text=text,
    )
