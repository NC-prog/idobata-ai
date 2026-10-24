"""実機（2026-10-24 夜）の不具合の直し: 提案タグの漏れ・砕けた相槌・お知らせの件数のログ・感想の文で聞き分けない。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from src.config import SERVER_ROOT, DialogueConfig, NewsConfig
from src.dialogue import Dialogue, strip_tags
from src.news import NewsBook, save_record
from src.site_context import load_facility
from tests.test_wp11_unit import NOW, RULES, _rec
from tests.test_wp13_clarify import _conv, _say

FAC = load_facility("akiota", SERVER_ROOT / "context")


class _LLM:
    """chat_stream は決まった断片。chat（英字の言い直し）は呼ばれたことを覚えて、訳した文を返す。"""

    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.rephrased: list[str] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> str:
        self.rephrased.append(messages[-1]["content"])
        return "共有する提案"

    def chat_stream(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        yield from self.deltas


async def _reply(deltas: list[str]) -> tuple[list[str], Dialogue, _LLM]:
    llm = _LLM(deltas)
    d = Dialogue(llm, DialogueConfig(), facility=FAC, enabled_actions=("share", "refer", "lookup"))  # type: ignore[arg-type]
    return [s async for s in d.respond("こんばんは")], d, llm


@pytest.mark.parametrize("deltas", [
    ["お変わりありませんか？", "傘はお持ちですか？", "<propose:share>"],  # 単独の文（22:55:50 の形）
    ["そうですね。<propose:share>"],  # 文末
    ["<propose:share>そうですね。"],  # 文頭
    ["そうですね、<propose:share>ありがとうございます。"],  # 文中
    ["そうですね。", "<propose:", "share>"],  # 断片に分かれて届く
    ["そうですね。<propose: share >"],
    ["そうですね。<propose share>"],
    ["そうですね。<propose:share"],  # 閉じていない（「>」が無い）
    ["そうですね。<share>"],  # 行為の名前だけ
    ["そうですね。＜propose：share＞"],  # 全角
])
async def test_propose_tag_never_spoken_and_reaches_proposal(deltas: list[str]) -> None:
    said, d, llm = await _reply(deltas)
    assert not any("<" in s or "＜" in s or "propose" in s.lower() or "share" in s.lower() or "共有" in s for s in said), said
    assert d.last_proposal == "share"
    assert llm.rephrased == []  # 英字の言い直しに回らない（タグは検査より前に剥がす）


async def test_other_tags_are_stripped_before_ascii_check() -> None:
    said, d, llm = await _reply(["そうですね。<br>", "</answer>ありがとうございます。", "<tool_call name=x>"])
    assert said == ["そうですね。", "ありがとうございます。"] and llm.rephrased == [] and d.last_proposal is None


def test_strip_tags_keeps_japanese_brackets() -> None:
    assert strip_tags("〈お知らせ〉3<5 のとき<propose:refer>")[0] == "〈お知らせ〉3<5 のとき"
    assert strip_tags("<propose:refer>")[1] == ["refer"]


@pytest.mark.parametrize(("raw", "spoken"), [
    ("おっ、お店を始めたいのですね。", "お店を始めたいのですね。"),
    ("おお、それはいいですね。", "それはいいですね。"),
    ("へぇ、そうなんですね。", "そうなんですね。"),
    ("あら、大変でしたね。", "大変でしたね。"),
    ("おおきな公園があります。", "おおきな公園があります。"),
    ("あらためてご案内します。", "あらためてご案内します。"),
])
async def test_casual_interjection_removed_at_head(raw: str, spoken: str) -> None:
    said, _, _ = await _reply([raw])
    assert said == [spoken]


async def test_casual_interjection_only_sentence_is_not_spoken() -> None:
    said, _, _ = await _reply(["へぇー。", "そうなんですね。"])
    assert said == ["そうなんですね。"]


def test_news_count_log_only_when_changed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="src.news")
    save_record(tmp_path, _rec("a"))
    book = NewsBook(tmp_path, NewsConfig(reload_sec=0), RULES, now=lambda: NOW)
    save_record(tmp_path, _rec("a", shown_count=1))  # 件数は同じで、ファイルが書き換わった
    book.reload(force=True)
    save_record(tmp_path, _rec("b"))  # 件数が変わった
    book.reload(force=True)
    lines = [(r.levelno, r.getMessage()) for r in caplog.records if r.getMessage().startswith("お知らせ:")]
    assert [lv for lv, _ in lines] == [logging.INFO, logging.DEBUG, logging.INFO]


@pytest.mark.parametrize("text", ["国保ってゴミだね", "国保は高いよね", "手当なんていらないでしょ", "保険証じゃない",
                                  "国保って面倒だよ", "マイナンバーカードはいらない"])
async def test_no_clarify_on_remarks(tmp_path: Path, text: str) -> None:
    c = _conv(tmp_path, StubLLMs())
    said = await _say(c, text)
    assert not said or not said[0].endswith("どちらでしょう？（1. 国保に入る手続き　2. 国保をやめる手続きや保険証　3. 国保税）")
    assert c._pending is None or c._pending.action != "clarify", text


@pytest.mark.parametrize("text", ["国保のことで", "手当のことで", "保険証", "国保のこと教えて", "国保じゃない？"])
async def test_clarify_on_questions_and_requests(tmp_path: Path, text: str) -> None:
    c = _conv(tmp_path, StubLLMs())
    await _say(c, text)
    assert c._pending is not None and c._pending.action == "clarify", text


def StubLLMs() -> Any:
    from tests.test_actions_unit import StubLLM

    return StubLLM([["そうなんですね。"]])
