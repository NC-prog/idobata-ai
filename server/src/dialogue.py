"""最小対話（WP00 / WP01）。

1 発話 → LLM 1 応答。固定のシステムプロンプト（井戸端AIの最小ペルソナ）と
直近 N ターンの履歴だけを持つ。履歴はプロセス内のみで、ディスクには書かない
（構想メモ 5. 個人情報の扱い方針）。

WP01: LLM の応答をストリーミングで受け、文末（「。」「？」「！」・改行）で切って
1 文ずつ yield する。呼び出し側は 1 文目から合成・再生を始められる。

対話FSM（IDLE → NOTICE → GREET → LISTEN ⇄ TALK → ...）は WP03 で書き直す。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

from .config import DialogueConfig, config

logger = logging.getLogger(__name__)

# LLM が応答できなかったときに返す定型文（言語別）
_FALLBACK_REPLY: dict[str, str] = {
    "ja": "すみません、少し聞き取れませんでした。もう一度お話しいただけますか。",
    "en": "Sorry, I didn't quite catch that. Could you say it again?",
}

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ChatBackend(Protocol):
    """Dialogue が使う LLM 側のインターフェース（LLMClient 互換）。"""

    def chat_stream(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        """チャット補完をストリーミングで実行し、content の断片を順に返す。"""
        ...


class SentenceSplitter:
    """ストリーミングの断片を文に切る。

    文末記号（連続していればまとめて）と、その直後の閉じ括弧までを 1 文とする。
    改行も区切り（改行そのものは文に含めない）。ASCII の "." は直後が空白のときだけ区切る。
    最初の 1 つに限り、soft_terminators（読点）でも切る（前が min_chars 文字以上のとき）。
    """

    def __init__(
        self,
        terminators: str,
        closers: str,
        soft_terminators: str = "",
        min_chars: int = 0,
    ) -> None:
        self._terminators = set(terminators)
        self._closers = set(closers)
        self._soft = set(soft_terminators)
        self._min_chars = min_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        """断片を足し、確定した文を返す。"""
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            end, resume = cut
            sentence = self._buf[:end].strip()
            self._buf = self._buf[resume:]
            if sentence:
                out.append(sentence)
                self._emitted += 1
        return out

    def flush(self) -> str:
        """残り（文末記号なしで終わった最後の文）を返す。"""
        rest, self._buf = self._buf.strip(), ""
        return rest

    def _find_cut(self) -> tuple[int, int] | None:
        """(文の終わり, 次の文の開始) を返す。まだ確定できなければ None。"""
        buf = self._buf
        n = len(buf)
        for i, ch in enumerate(buf):
            if ch == "\n":
                return i, i + 1
            is_term = ch in self._terminators
            if not is_term and self._emitted == 0 and ch in self._soft:
                is_term = len(buf[:i].strip()) >= self._min_chars
            if ch == ".":
                if i + 1 >= n:
                    return None  # 次の文字を見るまで決められない
                is_term = buf[i + 1].isspace()
            if not is_term:
                continue
            j = i + 1
            while j < n and (buf[j] in self._terminators or buf[j] in self._closers):
                j += 1
            if j >= n:
                return None  # 「？！」や「。」」の続きが来るかもしれない
            return j, j
        return None


class _ThinkFilter:
    """応答の先頭に <think>...</think> が来たら取り除く（思考モードは既定オフだが念のため）。"""

    def __init__(self) -> None:
        self._head = ""
        self._passing = False

    def feed(self, delta: str) -> str:
        if self._passing:
            return delta
        self._head += delta
        stripped = self._head.lstrip()
        if _THINK_OPEN.startswith(stripped):
            return ""  # "<thi" まで来た段階かもしれない
        if stripped.startswith(_THINK_OPEN):
            close = stripped.find(_THINK_CLOSE)
            if close < 0:
                return ""
            logger.debug("応答先頭の思考過程を破棄しました（%d文字）", close)
            stripped = stripped[close + len(_THINK_CLOSE):]
        self._passing = True
        return stripped


_END = object()


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
        # LLM ストリームを読むスレッド（タスクが GC で消えないよう参照を持つ）
        self._worker: asyncio.Future[None] | None = None

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

    async def respond(self, text: str, language: str = "ja") -> AsyncIterator[str]:
        """1 発話に対する応答を 1 文ずつ返す。

        Args:
            text: 来訪者の発話（ASR の確定テキスト、または text モードの入力）
            language: 検出言語（ISO 639-1）

        Yields:
            応答の文。LLM が 1 文も返せなかった場合は定型の聞き返し文を 1 つ。
            履歴には、言い終えた全文を 1 ターンとして積む。
        """
        text = text.strip()
        if not text:
            return

        messages = self.build_messages(text, language)
        logger.debug("発話 [%s]: %s", language, text)
        splitter = SentenceSplitter(
            self._cfg.sentence_terminators,
            self._cfg.sentence_closers,
            self._cfg.first_chunk_soft_terminators,
            self._cfg.first_chunk_min_chars,
        )
        think = _ThinkFilter()
        sentences: list[str] = []
        try:
            async for delta in self._stream(messages):
                for sentence in splitter.feed(think.feed(delta)):
                    sentences.append(sentence)
                    yield sentence
        except Exception:
            logger.exception("LLM 呼び出しに失敗しました")
            if not sentences:
                yield _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])
                return
            # 言いかけの残りは捨て、言えた文だけを履歴に積む
        else:
            tail = splitter.flush()
            if tail:
                sentences.append(tail)
                yield tail

        if not sentences:
            yield _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])
            return
        reply = "".join(sentences)
        self._history.append((text, reply))
        logger.debug("応答: %s", reply)

    async def _stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        """同期のストリーミング API を別スレッドで回し、断片を非同期に受け取る。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()

        def put(item: object) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass  # イベントループが先に閉じた

        def worker() -> None:
            try:
                for delta in self._llm.chat_stream(
                    self._model,
                    messages,
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": self._cfg.enable_thinking},
                    },
                ):
                    put(delta)
            except BaseException as exc:  # noqa: BLE001 — イベントループ側で送出し直す
                put(exc)
            else:
                put(_END)

        # 受け手が途中でやめても、スレッドはストリームを読み切って終わる
        self._worker = asyncio.ensure_future(asyncio.to_thread(worker))
        while True:
            item = await queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, str)
            yield item
