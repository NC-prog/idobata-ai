"""最小対話（WP00）。

1 発話 → LLM 1 応答。固定のシステムプロンプト（井戸端AIの最小ペルソナ）と
直近 N ターンの履歴だけを持つ。履歴はプロセス内のみで、ディスクには書かない
（構想メモ 5. 個人情報の扱い方針）。

対話FSM（IDLE → NOTICE → GREET → LISTEN ⇄ TALK → ...）は WP03 で書き直す。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Protocol

from .config import DialogueConfig, config

logger = logging.getLogger(__name__)

# LLM が応答できなかったときに返す定型文（言語別）
_FALLBACK_REPLY: dict[str, str] = {
    "ja": "すみません、少し聞き取れませんでした。もう一度お話しいただけますか。",
    "en": "Sorry, I didn't quite catch that. Could you say it again?",
}


class ChatBackend(Protocol):
    """Dialogue が使う LLM 側のインターフェース（LLMClient 互換）。"""

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """チャット補完を実行し、応答テキストを返す。"""
        ...


class Dialogue:
    """固定ペルソナ + 直近履歴での 1 発話 1 応答。"""

    def __init__(
        self,
        llm: ChatBackend,
        cfg: DialogueConfig | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._cfg = cfg or config.dialogue
        self._model = model or config.llm.chat_model
        # (user, assistant) の組を直近 history_turns 件だけ保持する
        self._history: deque[tuple[str, str]] = deque(maxlen=self._cfg.history_turns)

    @property
    def history_len(self) -> int:
        """保持している履歴ターン数。"""
        return len(self._history)

    def reset(self) -> None:
        """履歴を消す（来訪者の切り替わり時など）。"""
        self._history.clear()
        logger.info("対話履歴をクリアしました")

    def build_messages(self, text: str, language: str) -> list[dict[str, Any]]:
        """LLM に渡す messages を組み立てる。"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._cfg.system_prompt},
        ]
        for user, assistant in self._history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        content = text if language == "ja" else f"[language: {language}] {text}"
        messages.append({"role": "user", "content": content})
        return messages

    async def respond(self, text: str, language: str = "ja") -> str:
        """1 発話に対する応答を返す。

        Args:
            text: 来訪者の発話（ASR の確定テキスト、または text モードの入力）
            language: 検出言語（ISO 639-1）

        Returns:
            応答テキスト。LLM が失敗した場合は定型の聞き返し文。
        """
        text = text.strip()
        if not text:
            return ""

        messages = self.build_messages(text, language)
        logger.debug("発話 [%s]: %s", language, text)
        try:
            reply = await asyncio.to_thread(
                self._llm.chat,
                self._model,
                messages,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self._cfg.enable_thinking},
                },
            )
        except Exception:
            logger.exception("LLM 呼び出しに失敗しました")
            return _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])

        reply = reply.strip()
        if not reply:
            return _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])

        self._history.append((text, reply))
        logger.debug("応答: %s", reply)
        return reply


_default: Dialogue | None = None


async def respond(text: str, language: str = "ja") -> str:
    """モジュールレベルの既定 Dialogue で応答する（簡易 API）。"""
    global _default
    if _default is None:
        from .llm_client import LLMClient

        _default = Dialogue(LLMClient())
    return await _default.respond(text, language)
