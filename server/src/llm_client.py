"""vLLM Gateway クライアント。

OpenAI互換APIを通じて gpt-oss / deepseek-r1 を呼び出す。
Gateway (8000) が応答しない場合は直接ポートにフォールバックする。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ValidationError

from .config import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# JSON を本文から抽出する正規表現（```json ... ``` やプレーンテキストの両方に対応）
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# deepseek-r1 が content 内に混入させる思考過程タグ
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# 開始タグなしで </think> だけ出現するケース（先頭から </think> までが思考過程）
_THINK_CLOSE_ONLY_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


class LLMClient:
    """vLLM Gateway を利用する LLM クライアント。"""

    def __init__(self) -> None:
        cfg = config.llm
        self._gateway = OpenAI(base_url=cfg.gateway_url, api_key=cfg.api_key)
        self._direct_clients: dict[str, OpenAI] = {
            model: OpenAI(base_url=url, api_key=cfg.api_key)
            for model, url in cfg.direct_urls.items()
        }
        self._cfg = cfg

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """チャット補完を実行し、応答テキストを返す。

        Gateway → 直接ポートの順にフォールバックし、
        各エンドポイントで指数バックオフのリトライを行う。
        deepseek の reasoning_content は破棄する。
        """
        timeout = (
            self._cfg.reasoning_timeout
            if model == self._cfg.reasoning_model
            else self._cfg.timeout
        )
        kwargs.setdefault("timeout", timeout)
        max_tokens = (
            self._cfg.reasoning_max_tokens
            if model == self._cfg.reasoning_model
            else self._cfg.max_tokens
        )
        kwargs.setdefault("max_tokens", max_tokens)

        clients = self._clients_for(model)
        last_exc: Exception | None = None

        for label, client in clients:
            try:
                return self._chat_with_retry(client, model, messages, label, **kwargs)
            except (APIConnectionError, APIStatusError, APITimeoutError) as exc:
                logger.warning("%s への接続に失敗: %s", label, exc)
                last_exc = exc

        raise RuntimeError(
            f"すべてのエンドポイントで {model} への接続に失敗しました"
        ) from last_exc

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """JSON モードでチャット補完を実行し、Pydantic モデルとして返す。

        パース失敗時はエラー内容を添えて1回だけ修復リトライする。
        """
        raw = self.chat(model, messages, **kwargs)
        parsed = self._extract_json(raw)

        try:
            return schema.model_validate_json(parsed)
        except (ValidationError, json.JSONDecodeError) as first_err:
            logger.warning("JSON パース失敗（修復リトライ）: %s", first_err)
            err_detail = str(first_err)

        # 修復リトライ: エラー内容を添えて再生成
        repair_messages = [
            *messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"上記の応答を正しいJSONに修正してください。"
                    f"エラー: {err_detail}\n"
                    f"JSONのみを出力してください。"
                ),
            },
        ]
        raw_retry = self.chat(model, repair_messages, **kwargs)
        parsed_retry = self._extract_json(raw_retry)
        return schema.model_validate_json(parsed_retry)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _clients_for(self, model: str) -> list[tuple[str, OpenAI]]:
        """Gateway → 直接ポートの順にクライアントリストを返す。"""
        result: list[tuple[str, OpenAI]] = [("gateway", self._gateway)]
        if model in self._direct_clients:
            result.append((f"direct({model})", self._direct_clients[model]))
        return result

    def _chat_with_retry(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, Any]],
        label: str,
        **kwargs: Any,
    ) -> str:
        """指数バックオフ付きリトライ。"""
        timeout = kwargs.pop("timeout", self._cfg.timeout)
        last_exc: Exception | None = None

        for attempt in range(self._cfg.max_retries):
            if attempt > 0:
                wait = 2**attempt
                logger.info("リトライ %d/%d（%s、%d秒待機）",
                            attempt + 1, self._cfg.max_retries, label, wait)
                time.sleep(wait)

            t0 = time.monotonic()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    timeout=timeout,
                    **kwargs,
                )
                elapsed = time.monotonic() - t0
                usage = resp.usage
                tokens = f"{usage.prompt_tokens}+{usage.completion_tokens}" if usage else "?"
                logger.info(
                    "LLM応答 [%s] model=%s elapsed=%.1fs tokens=%s",
                    label, model, elapsed, tokens,
                )
                return self._extract_content(resp, model)

            except (APIConnectionError, APIStatusError, APITimeoutError) as exc:
                last_exc = exc
                logger.warning("試行 %d/%d 失敗 [%s]: %s",
                               attempt + 1, self._cfg.max_retries, label, exc)

        raise last_exc  # type: ignore[misc]

    def _extract_content(self, resp: Any, model: str) -> str:
        """応答から content を抽出する。deepseek の reasoning を完全に除去する。

        除去対象:
        1. reasoning_content 属性（OpenAI SDK が分離した思考過程）
        2. content 内の ``<think>...</think>`` タグ（SDK で分離されなかった場合）
        """
        choice = resp.choices[0]
        message = choice.message

        # 1. reasoning_content 属性の破棄
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            logger.debug("reasoning_content を破棄しました（%d文字）",
                         len(message.reasoning_content))

        content: str = message.content or ""

        # 2. content 内の思考過程タグを除去
        #    パターンA: <think>...</think> が揃っている場合
        if "<think>" in content:
            content = _THINK_RE.sub("", content)
        #    パターンB: 開始タグなしで </think> だけ出現する場合
        #    （deepseek-r1 が英語で思考し、閉じタグだけ出力する既知の挙動）
        if "</think>" in content:
            content = _THINK_CLOSE_ONLY_RE.sub("", content)

        if content != (message.content or ""):
            removed_len = len(message.content or "") - len(content)
            logger.info(
                "content 内の思考過程を除去しました（%d文字）", removed_len,
            )

        if not content.strip():
            logger.warning(
                "LLM から空の content が返されました（model=%s, "
                "reasoning_content=%d文字, finish_reason=%s）",
                model,
                len(getattr(message, "reasoning_content", "") or ""),
                getattr(choice, "finish_reason", "unknown"),
            )
        return content

    @staticmethod
    def _extract_json(text: str) -> str:
        """テキストから最初の JSON オブジェクトを抽出する。"""
        m = _JSON_RE.search(text)
        if m:
            return m.group(0)
        return text
