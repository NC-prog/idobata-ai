"""dialogue.py のテスト（LLM はスタブ）。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from src.config import DialogueConfig
from src.dialogue import Dialogue, SentenceSplitter, _ThinkFilter


class StubLLM:
    """chat_stream で決まった断片を返す。fail_after 個返したあと例外を投げる。"""

    def __init__(
        self,
        deltas: list[str] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.deltas = deltas if deltas is not None else ["そうなん", "ですね。"]
        self.fail_after = fail_after
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    def chat_stream(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        self.calls.append((model, messages, kwargs))
        for i, d in enumerate(self.deltas):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("boom")
            yield d
        if self.fail_after is not None and self.fail_after >= len(self.deltas):
            raise RuntimeError("boom")


async def _collect(d: Dialogue, text: str, language: str = "ja") -> list[str]:
    return [s async for s in d.respond(text, language)]


async def test_respond_uses_system_prompt_and_disables_thinking() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig(), model="/models/llm")
    assert await _collect(d, "こんにちは") == ["そうなんですね。"]
    model, messages, kwargs = llm.calls[0]
    assert model == "/models/llm"
    assert messages[0]["role"] == "system"
    assert "聞き役" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "こんにちは"}
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


async def test_yields_sentence_by_sentence_and_keeps_full_reply_in_history() -> None:
    llm = StubLLM(["こんにちは。今日は", "寒いですね？", "お体", "に気をつけて"])
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "やあ") == ["こんにちは。", "今日は寒いですね？", "お体に気をつけて"]
    await _collect(d, "次")
    _, messages, _ = llm.calls[-1]
    assert messages[2] == {
        "role": "assistant", "content": "こんにちは。今日は寒いですね？お体に気をつけて",
    }


async def test_history_keeps_only_recent_turns() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig(history_turns=6))
    for i in range(10):
        await _collect(d, f"発話{i}")
    assert d.history_len == 6
    _, messages, _ = llm.calls[-1]
    # system + 6 ターン × 2 + 今回の user
    assert len(messages) == 1 + 12 + 1
    assert messages[1]["content"] == "発話3"


async def test_non_japanese_is_tagged() -> None:
    llm = StubLLM(["Hello!"])
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "hello", language="en") == ["Hello!"]
    _, messages, _ = llm.calls[0]
    assert messages[-1]["content"] == "[language: en] hello"


async def test_llm_failure_returns_fallback_without_history() -> None:
    d = Dialogue(StubLLM(fail_after=0), DialogueConfig())
    reply = await _collect(d, "こんにちは")
    assert len(reply) == 1
    assert "もう一度" in reply[0]
    assert d.history_len == 0


async def test_failure_midway_keeps_spoken_sentences_only() -> None:
    llm = StubLLM(["一文目です。二文", "目の途中"], fail_after=2)
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "こんにちは") == ["一文目です。"]
    await _collect(d, "次")
    assert llm.calls[-1][1][2]["content"] == "一文目です。"


async def test_empty_reply_returns_fallback() -> None:
    d = Dialogue(StubLLM(["  ", "\n"]), DialogueConfig())
    reply = await _collect(d, "こんにちは")
    assert "もう一度" in reply[0]
    assert d.history_len == 0


async def test_empty_input_is_ignored() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "   ") == []
    assert llm.calls == []


async def test_reset_clears_history() -> None:
    d = Dialogue(StubLLM(), DialogueConfig())
    await _collect(d, "a")
    d.reset()
    assert d.history_len == 0


async def test_leading_think_block_is_dropped() -> None:
    d = Dialogue(StubLLM(["<thi", "nk>\n考え中</th", "ink>\n\nはい。"]), DialogueConfig())
    assert await _collect(d, "a") == ["はい。"]


# ------------------------------------------------------------------
# SentenceSplitter
# ------------------------------------------------------------------


def _split(deltas: list[str]) -> list[str]:
    cfg = DialogueConfig()
    sp = SentenceSplitter(cfg.sentence_terminators, cfg.sentence_closers)
    out: list[str] = []
    for d in deltas:
        out += sp.feed(d)
    tail = sp.flush()
    return out + ([tail] if tail else [])


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        (["あ。い。う"], ["あ。", "い。", "う"]),
        (["本当", "ですか？！", "はい"], ["本当ですか？！", "はい"]),
        (["「どうぞ。", "」と言った。"], ["「どうぞ。」", "と言った。"]),
        (["一行目\n\n二行目"], ["一行目", "二行目"]),
        (["Hello. ", "How are", " you? Fine."], ["Hello.", "How are you?", "Fine."]),
        (["It costs 1.5 yen."], ["It costs 1.5 yen."]),
        (["。"], ["。"]),
    ],
)
def test_sentence_splitter(deltas: list[str], expected: list[str]) -> None:
    assert _split(deltas) == expected


def test_splitter_waits_for_possible_continuation() -> None:
    cfg = DialogueConfig()
    sp = SentenceSplitter(cfg.sentence_terminators, cfg.sentence_closers)
    # 末尾の「？」の後に「！」が続くかもしれないので、まだ確定しない
    assert sp.feed("ですか？") == []
    assert sp.feed("！次") == ["ですか？！"]


def test_first_sentence_is_also_cut_at_comma() -> None:
    cfg = DialogueConfig()
    sp = SentenceSplitter(
        cfg.sentence_terminators, cfg.sentence_closers,
        cfg.first_chunk_soft_terminators, cfg.first_chunk_min_chars,
    )
    out = sp.feed("はい、住民票の写しは、二階の市民課です。手続きは、窓口で。") + [sp.flush()]
    # 「はい、」は短いので切らない。2 文目以降は読点で切らない
    assert out == ["はい、住民票の写しは、", "二階の市民課です。", "手続きは、窓口で。"]


def test_think_filter_passes_plain_text() -> None:
    f = _ThinkFilter()
    assert f.feed("こん") == "こん"
    assert f.feed("<think>") == "<think>"  # 先頭以外のタグには触らない
