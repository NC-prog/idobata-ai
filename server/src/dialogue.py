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
import re
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime
from typing import Any, Protocol

from .config import DialogueConfig, config
from .knowledge import JST, Reference, SearchResult, Weather
from .persona import build_system_prompt, build_turn_message, source_labels
from .site_context import Facility

logger = logging.getLogger(__name__)

# LLM が応答できなかったときに返す定型文（言語別）。WP03 から外国語にも日本語で答える
# （多言語は site プロファイルで後日）ので、日本語の文だけを持つ
_FALLBACK_REPLY: dict[str, str] = {
    "ja": "すみません、少し聞き取れませんでした。もう一度お話しいただけますか。",
}

# 応答末尾の出典の印（[出典:1,3]）と、文中に紛れた [1] などの番号。読み上げ前に取り除く
_SOURCE_MARK_RE = re.compile(r"[\[［]\s*(?:出典\s*[:：]?)?\s*([0-9０-９]+(?:\s*[,，、]\s*[0-9０-９]+)*)\s*[\]］]")

# 印を取ったあとに残る句読点の前の空白
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \u3000]+([。、！？!?])")

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ChatBackend(Protocol):
    """Dialogue が使う LLM 側のインターフェース（LLMClient 互換）。"""

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """チャット補完（ストリーミングなし）。英字混じりの文の言い直しに使う。"""
        ...

    def chat_stream(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        """チャット補完をストリーミングで実行し、content の断片を順に返す。"""
        ...


class Searcher(Protocol):
    """知識パックの検索（KnowledgeBase / KnowledgeStore）。"""

    def search(self, query: str) -> SearchResult:
        """発話で参考資料を探す。"""
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


def _is_non_japanese_kanji(ch: str) -> bool:
    """CJK 統合漢字のうち、日本語の文字集合（cp932）に無いもの（「险」「这」など）。"""
    if not ("\u4e00" <= ch <= "\u9fff"):
        return False
    try:
        ch.encode("cp932")
    except UnicodeEncodeError:
        return True
    return False
_ZEN_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


class Dialogue:
    """固定ペルソナ + 直近履歴での 1 発話 1 応答。"""

    def __init__(
        self,
        llm: ChatBackend,
        cfg: DialogueConfig | None = None,
        model: str | None = None,
        facility: Facility | None = None,
        knowledge: Searcher | None = None,
        weather: Weather | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(JST),
    ) -> None:
        self._llm = llm
        self._cfg = cfg or config.dialogue
        self._model = model or config.llm.chat_model
        self._facility = facility
        self._kb = knowledge
        self._weather = weather
        self._now = now
        self._ascii_re = re.compile(rf"[A-Za-zＡ-Ｚａ-ｚ]{{{self._cfg.ascii_run_min},}}")
        self._allow_ascii = {a.lower() for a in (facility.allow_ascii if facility else ())}
        self._banned = tuple(self._cfg.banned_phrases) + (facility.outside_places if facility else ())
        self._llm_source_re = re.compile(self._cfg.llm_source_sentence_pattern)
        self._mentions_source_re = re.compile(r"(?:ホームページ|サイト|お知らせ).{0,8}載って")
        # 直近のターンの検索結果（計測・出典表示用）
        self.last_search_ms: float | None = None
        self.last_references: list[Reference] = []
        self.last_cited: list[Reference] = []
        # (user, assistant) の組を直近 history_turns 件だけ保持する
        self._history: deque[tuple[str, str]] = deque(maxlen=self._cfg.history_turns)
        # LLM ストリームを読むスレッド（タスクが GC で消えないよう参照を持つ）
        self._worker: asyncio.Future[None] | None = None

    def attach_knowledge(self, knowledge: Searcher | None) -> None:
        """知識パックを後から付ける（読み込みに数秒かかるので起動処理の中で行う）。"""
        self._kb = knowledge

    @property
    def history_len(self) -> int:
        """保持している履歴ターン数。"""
        return len(self._history)

    def reset(self) -> None:
        """履歴を消す（来訪者の切り替わり時など）。"""
        self._history.clear()
        logger.info("対話履歴をクリアしました")

    def build_messages(
        self, text: str, language: str, refs: list[Reference] | None = None
    ) -> list[dict[str, Any]]:
        """LLM に渡す messages を組み立てる（システムプロンプトは毎ターン作り直す）。"""
        now = self._now()
        weather = self._weather.lines(now) if self._weather is not None else []
        system = build_system_prompt(self._cfg, self._facility, now.date(), weather)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for user, assistant in self._history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        content = text if language == "ja" else f"[language: {language}] {text}"
        # 履歴には発話だけを積む（参考資料は今のターンにだけ付ける）
        messages.append({"role": "user",
                         "content": build_turn_message(self._cfg, self._facility, refs or [], content)})
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

        if language != "ja":
            # 多言語は後日。LLM に任せると日本語で中身を答えてしまうので、決まった文を返す
            self.last_references, self.last_cited, self.last_search_ms = [], [], None
            logger.info("日本語以外の発話（%s）には準備中の旨を返します", language)
            yield self._cfg.foreign_reply
            return

        refs = await self._search(text)
        messages = self.build_messages(text, language, refs)
        logger.debug("発話 [%s]: %s", language, text)
        splitter = SentenceSplitter(
            self._cfg.sentence_terminators,
            self._cfg.sentence_closers,
            self._cfg.first_chunk_soft_terminators,
            self._cfg.first_chunk_min_chars,
        )
        think = _ThinkFilter()
        sentences: list[str] = []
        cited: set[int] = set()
        try:
            async for delta in self._stream(messages):
                for raw in splitter.feed(think.feed(delta)):
                    sentence = await self._finish_sentence(raw, cited)
                    if sentence:
                        sentences.append(sentence)
                        yield sentence
        except Exception:
            logger.exception("LLM 呼び出しに失敗しました")
            if not sentences:
                yield _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])
                return
            # 言いかけの残りは捨て、言えた文だけを履歴に積む
        else:
            tail = await self._finish_sentence(splitter.flush(), cited)
            if tail:
                sentences.append(tail)
                yield tail
        self.last_cited = [refs[i - 1] for i in sorted(cited) if 0 < i <= len(refs)]
        already = any(self._mentions_source_re.search(x) for x in sentences)
        if self.last_cited and not already:
            # 出典の一言は、使った資料の出どころから決めて足す（「観光協会のホームページ」など）。
            # 応答がすでに「〇〇のホームページに載っています」と言っていれば重ねない
            line = self._cfg.source_sentence.format(
                labels=source_labels(self._cfg, self._facility, self.last_cited))
            sentences.append(line)
            yield line
        # 緊急連絡先の名前を言ったのに番号を言っていなければ、番号の文を足す（いちばん大事な事実なので）
        for line in self._missing_emergency_numbers("".join(sentences)):
            sentences.append(line)
            yield line

        if not sentences:
            yield _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])
            return
        reply = "".join(sentences)
        self._history.append((text, reply))
        logger.debug("応答: %s", reply)

    async def _search(self, text: str) -> list[Reference]:
        """知識パックを検索する（CPU で数十 ms。イベントループを止めないよう別スレッド）。"""
        self.last_references, self.last_cited, self.last_search_ms = [], [], None
        if self._kb is None:
            return []
        try:
            result = await asyncio.to_thread(self._kb.search, text)
        except Exception:
            logger.exception("知識パックの検索に失敗しました（参考資料なしで続けます）")
            return []
        self.last_search_ms = result.elapsed_ms
        self.last_references = result.references
        logger.info("知識パック検索: %d 件, %.0f ms", len(result.references), result.elapsed_ms)
        for r in result.references:
            logger.debug("  参考資料 %.3f %s", r.vector_score, r.title)
        return result.references

    async def _finish_sentence(self, raw: str, cited: set[int]) -> str:
        """読み上げる前の 1 文の後処理: 出典の印を取り除き、英字混じりなら言い直させる。"""
        def take(m: re.Match[str]) -> str:
            for n in re.split(r"[,，、\s]+", m.group(1)):
                if n:
                    cited.add(int(n.translate(_ZEN_DIGITS)))
            return ""

        sentence = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _SOURCE_MARK_RE.sub(take, raw)).strip()
        for old, new in self._cfg.leak_replacements:
            sentence = sentence.replace(old, new)
        if self._llm_source_re.search(sentence):
            return ""  # 出典の一言はこちらで足す
        if any(p in sentence for p in self._banned):
            logger.warning("言わないことにしている言い回しを含む文を読み上げずに捨てました")
            logger.debug("捨てた文: %s", sentence)
            return ""
        if not sentence or not self._has_foreign_word(sentence):
            return sentence
        logger.info("英字の混じった文を日本語で言い直させます")
        logger.debug("言い直し前: %s", sentence)
        try:
            fixed = await asyncio.to_thread(
                self._llm.chat,
                self._model,
                [
                    {"role": "system", "content": self._cfg.rephrase_prompt},
                    {"role": "user", "content": sentence},
                ],
                max_tokens=self._cfg.rephrase_max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            logger.warning("言い直しに失敗しました。この文は読み上げません。", exc_info=True)
            return ""
        fixed = _SOURCE_MARK_RE.sub("", fixed).strip()
        if not fixed or self._has_foreign_word(fixed):
            logger.warning("言い直しても英字が残ったため、この文は読み上げません")
            return ""
        return fixed

    def _missing_emergency_numbers(self, reply: str) -> list[str]:
        if self._facility is None:
            return []
        out: list[str] = []
        digits = re.sub(r"[^0-9]", "", reply.translate(_ZEN_DIGITS))
        for e in self._facility.emergency_lines:
            short = re.split(r"[（(]", e.name, maxsplit=1)[0]
            if short in reply and re.sub(r"[^0-9]", "", e.number) not in digits:
                out.append(self._cfg.emergency_number_sentence.format(name=short, number=e.number))
        return out

    def _has_foreign_word(self, sentence: str) -> bool:
        """英字の連なり（許可語以外）か、日本語の文字集合に無い漢字（中国語の簡体字など）があるか。"""
        if any(m.lower() not in self._allow_ascii for m in self._ascii_re.findall(sentence)):
            return True
        return any(_is_non_japanese_kanji(ch) for ch in sentence)

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
                    presence_penalty=self._cfg.presence_penalty,
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
