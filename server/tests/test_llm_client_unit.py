"""llm_client の単体テスト。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.llm_client import _THINK_CLOSE_ONLY_RE, _THINK_RE, LLMClient

# ------------------------------------------------------------------
# _THINK_RE パターンのテスト
# ------------------------------------------------------------------


class TestThinkTagRemoval:
    """content 内の <think>...</think> タグ除去のテスト。"""

    def test_removes_think_block(self) -> None:
        raw = "<think>これは思考過程です。</think>\n実際の回答テキスト"
        result = _THINK_RE.sub("", raw)
        assert result == "実際の回答テキスト"

    def test_removes_multiline_think_block(self) -> None:
        raw = (
            "<think>\nまず、ユーザーがタウンホールミーティングの司会者であることを"
            "知りました。\n発表内容を見ると、田中太郎氏が...\n</think>\n\n"
            "上期キャンペーンの成果を共有いただき、ありがとうございました。"
        )
        result = _THINK_RE.sub("", raw)
        assert result == "上期キャンペーンの成果を共有いただき、ありがとうございました。"

    def test_no_think_tag_unchanged(self) -> None:
        raw = "通常のテキスト。思考過程なし。"
        result = _THINK_RE.sub("", raw)
        assert result == raw

    def test_preserves_content_after_think(self) -> None:
        raw = (
            "<think>*まず、...\n"
            "最終的に、...\n"
            "</think>\n\n"
            "会議は成功裡に終了しました。今後のアクションに期待します。"
        )
        result = _THINK_RE.sub("", raw).strip()
        assert result == "会議は成功裡に終了しました。今後のアクションに期待します。"

    def test_removes_chinese_reasoning(self) -> None:
        """中国語の思考過程が混入するケース（deepseek-r1 の既知の挙動）。"""
        raw = (
            "<think>嗯，用户是一位主持人的角色，正在组织一个Town Hall Meeting。"
            "他需要生成一个简短的Closing Note，用日语写，1到2句话。</think>\n"
            "本日のミーティングでは、重要な成果が共有されました。"
        )
        result = _THINK_RE.sub("", raw).strip()
        assert result == "本日のミーティングでは、重要な成果が共有されました。"


# ------------------------------------------------------------------
# _THINK_CLOSE_ONLY_RE パターンのテスト（開始タグなしの orphan </think>）
# ------------------------------------------------------------------


class TestOrphanThinkCloseTag:
    """開始タグなしで </think> だけ出現するケースのテスト。"""

    def test_removes_english_reasoning_with_orphan_close(self) -> None:
        """deepseek-r1 が英語で思考し、閉じタグだけ出力する実際のパターン。"""
        raw = (
            "Alright, I need to figure out how to help the user generate "
            "a closing note for the townhall meeting.\n"
            "</think>\n\n"
            "本日のタウンホールミーティングでは、重要な成果が共有されました。"
        )
        result = _THINK_CLOSE_ONLY_RE.sub("", raw)
        assert "</think>" not in result
        assert "Alright" not in result
        assert "本日のタウンホール" in result

    def test_removes_multiline_orphan_reasoning(self) -> None:
        """複数行の思考過程 + orphan </think>。"""
        raw = (
            "First, let me analyze the presentation content.\n"
            "The presenter discussed campaign results.\n"
            "I should summarize the key points.\n"
            "</think>\n\n"
            "上期キャンペーンの成果を共有いただき、ありがとうございました。"
        )
        result = _THINK_CLOSE_ONLY_RE.sub("", raw)
        assert "</think>" not in result
        assert "First" not in result
        assert "上期キャンペーン" in result

    def test_no_orphan_close_unchanged(self) -> None:
        raw = "通常のテキスト。思考過程なし。"
        assert "</think>" not in raw  # guard: no orphan close tag present


# ------------------------------------------------------------------
# _extract_content のテスト
# ------------------------------------------------------------------


class TestExtractContent:
    """LLMClient._extract_content のテスト。"""

    @pytest.fixture()
    def client(self) -> LLMClient:
        return LLMClient.__new__(LLMClient)

    def _make_resp(
        self,
        content: str,
        reasoning_content: str | None = None,
    ) -> SimpleNamespace:
        message = SimpleNamespace(content=content)
        if reasoning_content is not None:
            message.reasoning_content = reasoning_content
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])

    def test_normal_content(self, client: LLMClient) -> None:
        resp = self._make_resp("正常な回答テキスト")
        result = client._extract_content(resp, "/models/llm")
        assert result == "正常な回答テキスト"

    def test_reasoning_content_attribute_discarded(self, client: LLMClient) -> None:
        resp = self._make_resp(
            content="回答テキスト",
            reasoning_content="これは思考過程です。",
        )
        result = client._extract_content(resp, "/models/llm")
        assert result == "回答テキスト"
        assert "思考過程" not in result

    def test_think_tag_in_content_removed(self, client: LLMClient) -> None:
        resp = self._make_resp(
            "<think>まず、ユーザーが...長い思考過程...</think>\n\n"
            "上期キャンペーンの成果を共有いただき、ありがとうございました。"
        )
        result = client._extract_content(resp, "/models/llm")
        assert "<think>" not in result
        assert "</think>" not in result
        assert "上期キャンペーン" in result

    def test_both_reasoning_sources_removed(self, client: LLMClient) -> None:
        """reasoning_content 属性と content 内の <think> タグが両方ある場合。"""
        resp = self._make_resp(
            content="<think>content内の思考</think>\n回答本文",
            reasoning_content="属性側の思考",
        )
        result = client._extract_content(resp, "/models/llm")
        assert result.strip() == "回答本文"

    def test_orphan_close_tag_removed(self, client: LLMClient) -> None:
        """開始タグなしで </think> だけ出現するケース（実際の障害パターン）。"""
        resp = self._make_resp(
            "Alright, I need to figure out how to help the user "
            "generate a closing note for the townhall meeting.\n"
            "The presenter discussed campaign results and Zscaler.\n"
            "</think>\n\n"
            "本日のタウンホールミーティングでは、重要な成果が共有されました。"
        )
        result = client._extract_content(resp, "/models/llm")
        assert "<think>" not in result
        assert "</think>" not in result
        assert "Alright" not in result
        assert "本日のタウンホール" in result

    def test_orphan_close_tag_with_reasoning_content(self, client: LLMClient) -> None:
        """reasoning_content 属性 + orphan </think> の両方がある場合。"""
        resp = self._make_resp(
            content=(
                "Let me think about this...\n</think>\n\n"
                "会議は成功裡に終了しました。"
            ),
            reasoning_content="SDK側の思考過程",
        )
        result = client._extract_content(resp, "/models/llm")
        assert result.strip() == "会議は成功裡に終了しました。"


# ------------------------------------------------------------------
# chat_stream（WP01）
# ------------------------------------------------------------------


def _chunk(content: str | None = None, usage: Any = None) -> SimpleNamespace:
    choices = [] if content is None else [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


class _FakeCompletions:
    def __init__(self, behaviours: list[Any]) -> None:
        self.behaviours = behaviours
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        b = self.behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return iter(b)


def _stream_client(behaviours: list[Any]) -> tuple[LLMClient, _FakeCompletions]:
    from src.config import LLMConfig

    comp = _FakeCompletions(behaviours)
    client = LLMClient.__new__(LLMClient)
    client._cfg = LLMConfig(max_retries=2)
    client._gateway = SimpleNamespace(chat=SimpleNamespace(completions=comp))  # type: ignore[assignment]
    client._direct_clients = {}
    return client, comp


class TestChatStream:
    def test_yields_deltas_and_requests_stream(self) -> None:
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3)
        client, comp = _stream_client([[_chunk("こん"), _chunk(""), _chunk("にちは"),
                                        _chunk(usage=usage)]])
        assert list(client.chat_stream("/models/llm", [])) == ["こん", "にちは"]
        assert comp.kwargs[0]["stream"] is True
        assert comp.kwargs[0]["stream_options"] == {"include_usage": True}

    def test_retries_before_first_delta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from openai import APIConnectionError

        monkeypatch.setattr("src.llm_client.time.sleep", lambda _s: None)
        err = APIConnectionError(request=httpx.Request("POST", "http://x"))
        client, comp = _stream_client([err, [_chunk("はい")]])
        assert list(client.chat_stream("/models/llm", [])) == ["はい"]
        assert len(comp.kwargs) == 2

    def test_failure_after_first_delta_is_raised(self) -> None:
        from openai import APIConnectionError

        err = APIConnectionError(request=httpx.Request("POST", "http://x"))

        def broken() -> Any:
            yield _chunk("途中")
            raise err

        client, comp = _stream_client([broken()])
        it = client.chat_stream("/models/llm", [])
        assert next(it) == "途中"
        with pytest.raises(APIConnectionError):
            next(it)
        assert len(comp.kwargs) == 1
