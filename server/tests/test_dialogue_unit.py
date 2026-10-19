"""dialogue.py のテスト（LLM はスタブ）。"""

from __future__ import annotations

from typing import Any

from src.config import DialogueConfig
from src.dialogue import Dialogue


class StubLLM:
    def __init__(self, reply: str = "そうなんですね。", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append((model, messages, kwargs))
        if self.fail:
            raise RuntimeError("boom")
        return self.reply


async def test_respond_uses_system_prompt_and_disables_thinking() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig(), model="/models/llm")
    reply = await d.respond("こんにちは")
    assert reply == "そうなんですね。"
    model, messages, kwargs = llm.calls[0]
    assert model == "/models/llm"
    assert messages[0]["role"] == "system"
    assert "聞き役" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "こんにちは"}
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


async def test_history_keeps_only_recent_turns() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig(history_turns=6))
    for i in range(10):
        await d.respond(f"発話{i}")
    assert d.history_len == 6
    _, messages, _ = llm.calls[-1]
    # system + 6 ターン × 2 + 今回の user
    assert len(messages) == 1 + 12 + 1
    assert messages[1]["content"] == "発話3"


async def test_non_japanese_is_tagged() -> None:
    llm = StubLLM("Hello!")
    d = Dialogue(llm, DialogueConfig())
    await d.respond("hello", language="en")
    _, messages, _ = llm.calls[0]
    assert messages[-1]["content"] == "[language: en] hello"


async def test_llm_failure_returns_fallback_without_history() -> None:
    d = Dialogue(StubLLM(fail=True), DialogueConfig())
    reply = await d.respond("こんにちは")
    assert "もう一度" in reply
    assert d.history_len == 0


async def test_empty_input_is_ignored() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig())
    assert await d.respond("   ") == ""
    assert llm.calls == []


async def test_reset_clears_history() -> None:
    d = Dialogue(StubLLM(), DialogueConfig())
    await d.respond("a")
    d.reset()
    assert d.history_len == 0
