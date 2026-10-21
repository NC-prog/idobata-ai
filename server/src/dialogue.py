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
from datetime import date, datetime, timedelta
from typing import Any, Protocol

from .config import DialogueConfig, config
from .knowledge import JST, Reference, SearchResult, Weather
from .persona import Persona, build_system_prompt, build_turn_message, source_labels, this_weekend
from .procedures import Procedures
from .site_context import Facility
from .timeofday import fix_greeting, greeting_for

logger = logging.getLogger(__name__)

# LLM が応答できなかったときに返す定型文（言語別）。WP03 から外国語にも日本語で答える
# （多言語は site プロファイルで後日）ので、日本語の文だけを持つ
_FALLBACK_REPLY: dict[str, str] = {
    "ja": "すみません、少し聞き取れませんでした。もう一度お話しいただけますか。",
}

# 応答末尾の出典の印（[出典:1,3]）と、文中に紛れた [1] などの番号。読み上げ前に取り除く
# 行為の提案のタグ（<propose:lookup> など）。読み上げ前に取り除き、last_proposal に残す
_PROPOSE_RE = re.compile(r"[<＜]\s*propose\s*[:：]\s*([a-z_]+)\s*[>＞]", re.IGNORECASE)

_SOURCE_MARK_RE = re.compile(
    r"[\[［]\s*(?:出典\s*[:：]?)?\s*([0-9０-９]+(?:\s*[,，、]\s*[0-9０-９]+)*)\s*[\]］]"
    # 括弧を付け忘れた印（「出典:1」「出典：1,3」）
    r"|[（(]?出典\s*[:：]\s*([0-9０-９]+(?:\s*[,，、]\s*[0-9０-９]+)*)[）)]?"
)

# 印を取ったあとに残る句読点の前の空白
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \u3000]+([。、！？!?])")
# 読み上げない括弧書き（全角・半角。入れ子は外側ごと）
_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]|[（(][^（）()]*[（(][^（）()]*[）)][^（）()]*[）)]")
# おうむ返しを捨てた後に残る、文の頭の助詞・つなぎ
_LEADING_PARTICLE_RE = re.compile(r"^(と聞かれると|とおっしゃると)[、,]?\s*|^(って|とは|は|が|を|に|も)[、,]\s*")
# 来訪者の言葉の繰り返しを比べるときに無視する文字
_ECHO_IGNORE_RE = re.compile(r"[\s、。，．,.！？!?「」『』…ー〜~]")

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
_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
# 年付きの日付（令和8年1月25日 / 2026年1月25日。令和元年も）
_DATED_RE = re.compile(r"(令和|R)?\s*(元|\d{1,4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")


def latest_dated(text: str) -> date | None:
    """本文に書かれた年付きの日付のうち、いちばん新しいもの。無ければ None。"""
    latest: date | None = None
    for era, y, m, d in _DATED_RE.findall(text.translate(_ZEN)):
        year = 1 if y == "元" else int(y)
        if era or year < 100:
            year += 2018  # 令和
        try:
            dt = date(year, int(m), int(d))
        except ValueError:
            continue
        latest = dt if latest is None or dt > latest else latest
    return latest


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


def speech_text(sentence: str) -> str:
    """読み上げる文（括弧書きは字幕だけに残し、読み上げない）。"""
    return re.sub(r"\s+", " ", _PAREN_RE.sub("", sentence)).strip()


def ends_with_closing(reply: str, cfg: DialogueConfig) -> bool:
    """応答の最後の文（こちらで足す出典の一言は除く）が締めの定型か。"""
    sentences = [x for x in re.split(r"(?<=[。！？!?])", reply) if x.strip()]
    source = re.compile(cfg.llm_source_sentence_pattern)
    while sentences and (source.search(sentences[-1].strip()) or "詳しくは" in sentences[-1]):
        sentences.pop()
    return bool(sentences) and re.search(cfg.closing_pattern, sentences[-1]) is not None


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
        avatar_name: str = "",
        enabled_actions: tuple[str, ...] = (),
        persona: Persona | None = None,
        procedures: Procedures | None = None,
    ) -> None:
        self._llm = llm
        self._cfg = cfg or config.dialogue
        self._model = model or config.llm.chat_model
        self._facility = facility
        self._kb = knowledge
        self._weather = weather
        self._avatar_name = avatar_name
        self._enabled_actions = enabled_actions
        self._persona = persona
        self._procedures = procedures
        # 直近の応答が出した行為の提案（<propose:…>）。無ければ None
        self.last_proposal: str | None = None
        self._now = now
        self._ascii_re = re.compile(rf"[A-Za-zＡ-Ｚａ-ｚ]{{{self._cfg.ascii_run_min},}}")
        self._allow_ascii = {a.lower() for a in (
            (facility.allow_ascii if facility else ()) + self._cfg.allow_ascii_common)}
        self._banned = tuple(self._cfg.banned_phrases) + (facility.outside_places if facility else ())
        self._llm_source_re = re.compile(self._cfg.llm_source_sentence_pattern)
        self._mentions_source_re = re.compile(r"(?:ホームページ|サイト|お知らせ).{0,8}載って")
        self._known_facilities = tuple(
            f.name for f in (facility.facilities_in_building if facility else ()))
        self._location_re = re.compile(self._cfg.building_location_pattern)
        self._facility_question_re = re.compile(self._cfg.facility_question_pattern)
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
        self, text: str, language: str, refs: list[Reference] | None = None, hint: str = ""
    ) -> list[dict[str, Any]]:
        """LLM に渡す messages を組み立てる（システムプロンプトは毎ターン作り直す）。"""
        now = self._now()
        weather = self._weather.lines(now) if self._weather is not None else []
        system = build_system_prompt(self._cfg, self._facility, now.date(), weather,
                                     self._avatar_name, self._enabled_actions,
                                     greeting_for(now, self._cfg), self._persona)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for user, assistant in self._history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        content = text if language == "ja" else f"[language: {language}] {text}"
        # 履歴には発話だけを積む（参考資料は今のターンにだけ付ける）
        # hint は今のターンにだけ添える材料（さっき来た方の小ネタなど）
        messages.append({"role": "user",
                         "content": hint + build_turn_message(self._cfg, self._facility, refs or [], content)})
        return messages

    @property
    def history(self) -> list[tuple[str, str]]:
        """今の会話の (来訪者, 応答) の組（古い順）。"""
        return list(self._history)

    def extend_last_reply(self, line: str) -> None:
        """直前の応答の後ろに文を足す（行為の提案文など。履歴の流れを保つため）。"""
        if self._history:
            user, reply = self._history[-1]
            self._history[-1] = (user, reply + line)

    def add_turn(self, user: str, assistant: str) -> None:
        """LLM を通さずに答えたターン（行為の定型文など）を履歴に積む。"""
        self._history.append((user, assistant))

    async def respond(
        self,
        text: str,
        language: str = "ja",
        extra_refs: list[Reference] | None = None,
        hint: str = "",
    ) -> AsyncIterator[str]:
        """1 発話に対する応答を 1 文ずつ返す。

        Args:
            text: 来訪者の発話（ASR の確定テキスト、または text モードの入力）
            language: 検出言語（ISO 639-1）

        Yields:
            応答の文。LLM が 1 文も返せなかった場合は定型の聞き返し文を 1 つ。
            履歴には、言い終えた全文を 1 ターンとして積む。
        """
        self.last_proposal = None
        text = text.strip()
        if not text:
            return

        if language != "ja":
            # 多言語は後日。LLM に任せると日本語で中身を答えてしまうので、決まった文を返す
            self.last_references, self.last_cited, self.last_search_ms = [], [], None
            logger.info("日本語以外の発話（%s）には準備中の旨を返します", language)
            yield self._cfg.foreign_reply
            return

        asked = self._asked_unknown_facility(text)
        if asked is not None:
            # 登録の無い館内の設備の場所・有無は、場所を作らせないよう決まった文で答える
            self.last_references, self.last_cited, self.last_search_ms = [], [], None
            reply = self._cfg.unknown_facility_sentence.format(name=asked)
            logger.info("登録の無い館内の設備（%s）を聞かれたので、決まった文で答えます", asked)
            self._history.append((text, reply))
            yield reply
            return

        refs = list(extra_refs) if extra_refs is not None else await self._search(text)
        if extra_refs is not None:
            self.last_references, self.last_search_ms = refs, None
        place = self._unknown_place(text, refs)
        if place is not None:
            # 地区・町の様子を聞かれたのに資料が無い: 様子を作らず、教えてもらう（人物像の「土地勘がまだ薄い」）
            reply = self._cfg.unknown_place_reply.format(place=place)
            logger.info("資料の無い地区の様子を聞かれたので、教えてもらう文で返します")
            self._history.append((text, reply))
            yield reply
            return
        messages = self.build_messages(text, language, refs, hint)
        logger.debug("発話 [%s]: %s", language, text)
        splitter = SentenceSplitter(
            self._cfg.sentence_terminators,
            self._cfg.sentence_closers,
            self._cfg.first_chunk_soft_terminators,
            self._cfg.first_chunk_min_chars,
        )
        think = _ThinkFilter()
        # これまでの応答の文（同じ文を繰り返さない。短い返事の後に前の応答をそのまま言い直す癖がある）
        previous = {_ECHO_IGNORE_RE.sub("", x) for _, reply in self._history
                    for x in re.split(r"(?<=[。！？!?])", reply)}
        sentences: list[str] = []
        dropped_echo = False
        # 発話から担当課が規則で一意に決まるなら、応答で最初に名前を出す課をそれにそろえる
        expected_dept = self._expected_department(text)
        dept_state = expected_dept is None
        cited: set[int] = set()
        try:
            async for delta in self._stream(messages):
                for raw in splitter.feed(think.feed(delta)):
                    sentence = await self._finish_sentence(raw, cited)
                    sentence, dept_state = self._fix_department(sentence, expected_dept, dept_state)
                    sentence = self._fix_floor_of_last_department(sentence, sentences)
                    if self._drop(sentence, text, sentences, previous):
                        dropped_echo = dropped_echo or not sentences
                        continue
                    if dropped_echo and not sentences:
                        # おうむ返しの断片を捨てた後の続き（「は、保健医療課で…」）の頭の助詞を外す
                        sentence = _LEADING_PARTICLE_RE.sub("", sentence)
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
            tail, dept_state = self._fix_department(tail, expected_dept, dept_state)
            tail = self._fix_floor_of_last_department(tail, sentences)
            if tail and dropped_echo and not sentences:
                tail = _LEADING_PARTICLE_RE.sub("", tail)
            # 何も言っていないときの最後の文は捨てない（応答が空になる）
            if tail and (not sentences or not self._drop(tail, text, sentences, previous)):
                sentences.append(tail)
                yield tail
        self.last_cited = [refs[i - 1] for i in sorted(cited) if 0 < i <= len(refs)]
        already = any(self._mentions_source_re.search(x) for x in sentences)
        # 出典の一言は、応答が資料の中身を実際に使ったときだけ（雑談に関係の薄い資料の番号が付いても足さない）
        if self.last_cited and not already and self._uses_references(sentences, self.last_cited):
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

    def _rain_likely(self) -> bool:
        """天気の文の降水確率に、傘をすすめる基準以上のものがあるか（天気が無ければ判断しない＝True）。"""
        if self._weather is None:
            return True
        lines = self._weather.lines(self._now())
        pops = [int(x) for line in lines for x in re.findall(r"(\d+)パーセント", line)]
        return not pops or max(pops) >= self._cfg.umbrella_min_pop

    def _expected_department(self, text: str) -> str | None:
        if self._procedures is None or self._facility is None:
            return None
        cands = self._procedures.departments_for(text, self._facility)
        return cands[0] if len(cands) == 1 else None

    def _fix_floor_of_last_department(self, sentence: str, said: list[str]) -> str:
        """課の名前の無い「1階ロビーの窓口にあります」を、この応答で直前に出た課の階に直す。"""
        depts = {d.name: d.floor for d in (self._facility.departments if self._facility else ()) if d.floor}
        last = next((n for x in reversed(said) for n in depts if n in x), None)
        if last is None or any(n in sentence for n in depts):
            return sentence
        floor = depts[last]
        fixed = re.sub(r"[0-9０-９一二三四五]階(ロビー)?(の窓口)?(に|で)(あり|ござい)",
                       lambda m: m[0] if m[0].startswith(floor) else f"{floor}{m[3]}{m[4]}", sentence)
        if fixed != sentence:
            logger.info("課の階を施設の事実に直しました（%s）", last)
        return fixed

    def _fix_department(self, sentence: str, expected: str | None, done: bool) -> tuple[str, bool]:
        """応答で最初に出た課が、発話から決まる担当課と違えば直す。done はもう課を言い終えたか。

        2 つ目以降の課は直さない（「社会福祉課でも受け付けています」のような追加の案内があるため）。
        """
        if done or expected is None or not sentence:
            return sentence, done
        names = [d.name for d in self._facility.departments] if self._facility else []
        said = [n for n in names if n in sentence]
        if not said:
            return sentence, False
        if expected in said:
            return sentence, True
        wrong = said[0]
        logger.info("担当課の取り違えを直しました（%s → %s）", wrong, expected)
        floor = next((d.floor for d in self._facility.departments if d.name == expected), None) if self._facility else None
        fixed = sentence.replace(wrong, expected, 1)
        if floor:  # 「市民課（1階）」の階も直す
            fixed = self._fix_department_floor(fixed)
        return fixed, True

    def _fix_department_floor(self, sentence: str) -> str:
        """「1階の保健医療課」「税務課（2階）」「市民課は2階」の階を、施設の事実の階に直す。"""
        for d in (self._facility.departments if self._facility else ()):
            if not d.floor or d.name not in sentence:
                continue
            n, floor = re.escape(d.name), d.floor.replace("\\", "")
            fixed = re.sub(rf"[0-9０-９一二三四五]階(の|にある)({n})", rf"{floor}\g<1>\g<2>", sentence)
            fixed = re.sub(rf"({n})(（|\(|は|が)[0-9０-９一二三四五]階", rf"\g<1>\g<2>{floor}", fixed)
            if fixed != sentence:
                logger.info("課の階を施設の事実に直しました（%s）", d.name)
                sentence = fixed
        return sentence

    def _department_directions(self, sentence: str) -> bool:
        """館内の道順の言い方を含む文か（道順は作らない決まり。課の名前が前の文にあっても捕まえる）。"""
        return re.search(self._cfg.direction_pattern, sentence) is not None

    def _unknown_place(self, text: str, refs: list[Reference]) -> str | None:
        """「〇〇ってどんなところ？」で参考資料が無ければ、その地名（無ければ「そちら」）。"""
        if refs:
            return None
        m = re.search(self._cfg.place_question_pattern, text)
        if m is None:
            return None
        return m.group("place").strip("、。 ") or "そちら"

    def _contradicts_persona(self, sentence: str) -> bool:
        return self._persona is not None and any(c in sentence for c in self._persona.contradictions)

    def _unasked_self_intro(self, sentence: str, user_text: str) -> bool:
        name = self._avatar_name
        if not name or re.search(self._cfg.name_question_pattern, user_text):
            return False
        return re.fullmatch(rf"(はい、)?{re.escape(name)}(です|と申します|といいます)[。！!]?", sentence.strip()) is not None

    def _uses_references(self, sentences: list[str], refs: list[Reference]) -> bool:
        reply = _ECHO_IGNORE_RE.sub("", "".join(sentences))
        body = _ECHO_IGNORE_RE.sub("", "".join(f"{r.title}{r.text}" for r in refs))
        n = self._cfg.source_overlap_ngram
        grams = {reply[i:i + n] for i in range(len(reply) - n + 1)}
        if not grams:
            return False
        hit = sum(1 for g in grams if g in body) / len(grams)
        if hit < self._cfg.source_min_overlap:
            logger.info("応答が資料の中身をほとんど使っていないので、出典の一言は足しません（%.2f）", hit)
            return False
        return True

    def _drop(self, sentence: str, text: str, said: list[str], previous: set[str]) -> bool:
        """読み上げない文か（おうむ返し・これまでと同じ文・重ねた「お困りのことは」の類）。"""
        if not sentence:
            return False
        if self._history and re.fullmatch(r"(おはようございます|こんにちは|こんばんは)[。！!、]?", sentence.strip()):
            logger.info("会話の途中の挨拶だけの文を読み上げずに捨てました")
            return True
        if "傘" in sentence and not self._rain_likely():
            logger.info("降水確率が低いのに傘をすすめる文を読み上げずに捨てました")
            return True
        if self._contradicts_persona(sentence):
            logger.info("人物像と食い違う文（生まれ・育ちなど）を読み上げずに捨てました")
            return True
        if self._unasked_self_intro(sentence, text):
            logger.info("聞かれていないのに名乗る文を読み上げずに捨てました")
            return True
        if not said and self._is_echo(sentence, text):
            logger.info("来訪者の言葉を繰り返す文を読み上げずに捨てました")
            return True
        if self._is_repeat(sentence, previous):
            logger.info("これまでの応答と同じ文を読み上げずに捨てました")
            return True
        if self._extra_help_offer(sentence, text):
            logger.info("「お困りのことは」「お変わりありませんか」の類を重ねない（1 会話 1 回・尋ねられた直後は言わない）")
            return True
        return False

    def _extra_help_offer(self, sentence: str, user_text: str) -> bool:
        """「お困りのことはありませんか」「お変わりありませんか」の類を言わない場面か（もう言った、または尋ねられた直後）。"""
        for pat in (self._cfg.help_offer_pattern, self._cfg.checkin_pattern):
            if not re.search(pat, sentence):
                continue
            if re.search(self._cfg.visitor_question_pattern, user_text.strip()):
                return True
            return any(re.search(pat, reply) for _, reply in self._history)
        return False

    def _is_repeat(self, sentence: str, previous: set[str]) -> bool:
        s = _ECHO_IGNORE_RE.sub("", sentence)
        return len(s) >= self._cfg.repeat_min_chars and s in previous

    def _is_echo(self, sentence: str, user_text: str) -> bool:
        """応答の文が来訪者の言葉のおうむ返しか（挨拶を返すのは除く）。"""
        if any(k in sentence for k in self._cfg.echo_keep):
            return False
        s = _ECHO_IGNORE_RE.sub("", sentence)
        u = _ECHO_IGNORE_RE.sub("", user_text)
        if not s or s not in u:
            return False
        return len(s) >= 4 or u.startswith(s)

    async def _search(self, text: str) -> list[Reference]:
        """知識パックを検索する（CPU で数十 ms。イベントループを止めないよう別スレッド）。"""
        self.last_references, self.last_cited, self.last_search_ms = [], [], None
        if self._kb is None:
            return []
        query = self._expand_dates(text)
        try:
            result = await asyncio.to_thread(self._kb.search, query)
        except Exception:
            logger.exception("知識パックの検索に失敗しました（参考資料なしで続けます）")
            return []
        self.last_search_ms = result.elapsed_ms
        refs = result.references
        if query != text:
            # 日付を聞かれたときは、書かれた日付（年付き）がすべて過ぎている資料を外す
            # （1月の「とんど祭り」を今週末の行事として案内したため）
            today = self._now().date()
            past = [r for r in refs if (d := latest_dated(r.text)) is not None and d < today]
            if past:
                logger.info("日付の過ぎた資料を %d 件外しました", len(past))
                refs = [r for r in refs if r not in past]
        self.last_references = refs
        logger.info("知識パック検索: %d 件, %.0f ms", len(refs), result.elapsed_ms)
        for r in refs:
            logger.debug("  参考資料 %.3f %s", r.vector_score, r.title)
        return refs

    def _expand_dates(self, text: str) -> str:
        """「今週末」「来週末」「今日」「明日」を具体的な日付にして検索語に足す（行事の資料は日付で書かれている）。"""
        today = self._now().date()
        sat, sun = this_weekend(today)
        week = timedelta(days=7)
        spans: list[date] = []
        if "今週末" in text or "この週末" in text or "週末" == text.strip("はに？?、。 "):
            spans += [sat, sun]
        if "来週末" in text:
            spans += [sat + week, sun + week]
        if "今日" in text:
            spans.append(today)
        if "明日" in text:
            spans.append(today + timedelta(days=1))
        if not spans:
            return text
        # 「1月25日」の行事に「10月25日」が部分一致しないよう、令和の年と月も足す（市町の記事は和暦で書く）
        months = dict.fromkeys(f"令和{d.year - 2018}年{d.month}月" for d in spans)
        return text + " " + " ".join([f"{d.month}月{d.day}日" for d in spans] + list(months))

    async def _finish_sentence(self, raw: str, cited: set[int]) -> str:
        """読み上げる前の 1 文の後処理: 出典の印を取り除き、英字混じりなら言い直させる。"""
        def take(m: re.Match[str]) -> str:
            for n in re.split(r"[,，、\s]+", m.group(1) or m.group(2) or ""):
                if n:
                    cited.add(int(n.translate(_ZEN_DIGITS)))
            return ""

        def propose(m: re.Match[str]) -> str:
            kind = m.group(1).lower()
            if kind in self._enabled_actions:
                self.last_proposal = kind
            return ""

        raw = _PROPOSE_RE.sub(propose, raw)
        sentence = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", _SOURCE_MARK_RE.sub(take, raw)).strip()
        sentence = fix_greeting(sentence, greeting_for(self._now(), self._cfg), self._cfg)
        if self._avatar_name:  # 「いどちゃん: こんばんは」のような話者名は外す
            # 「いどちゃん: …」「いどちゃんは、…」「いどちゃん物忘れが…」の頭の名前を外す（「いどちゃんです」は残す）
            name = re.escape(self._avatar_name)
            sentence = re.sub(rf"^{name}\s*(?:[:：、]|[はも]、)\s*|^{name}(?=[一-龥ァ-ヶ])", "", sentence)
        for old, new in self._cfg.leak_replacements:
            sentence = sentence.replace(old, new)
        if self._llm_source_re.search(sentence):
            return ""  # 出典の一言はこちらで足す
        sentence = self._fix_department_floor(sentence)
        unknown = self._unknown_facility_location(sentence)
        if unknown is not None:
            logger.info("館内の設備（%s）の場所を作った文を差し替えます", unknown)
            logger.debug("差し替え前: %s", sentence)
            return self._cfg.unknown_facility_sentence.format(name=unknown)
        if self._department_directions(sentence):
            logger.info("館内の道順を作った文を読み上げずに捨てました")
            return ""
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

    def _is_known_facility(self, word: str) -> bool:
        return any(word in known or known in word for known in self._known_facilities)

    def _asked_unknown_facility(self, text: str) -> str | None:
        """来訪者が登録の無い館内の設備の場所・有無を聞いているなら、その設備名。"""
        if not self._facility_question_re.search(text):
            return None
        for word in self._cfg.building_facility_words:
            if word in text and not self._is_known_facility(word):
                return word
        return None

    def _unknown_facility_location(self, sentence: str) -> str | None:
        """登録されていない館内の設備の場所を言っている文なら、その設備名。"""
        for word in self._cfg.building_facility_words:
            if word not in sentence:
                continue
            if self._is_known_facility(word):
                continue  # 登録済み（AED など）は、固定事実どおりに言ってよい
            # 「近くの職員」は場所の言い方ではない
            rest = sentence.replace("近くの職員", "")
            if self._location_re.search(rest):
                return word
        return None

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

    def safe_query(self, raw: str | None) -> str | None:
        """LLM が作った検索語の安全装置（英字・言わない言い回し・長すぎるものは使わない）。"""
        q = re.sub(r"[「」『』、。,.!?！？\n]", " ", (raw or "")).strip()
        q = re.sub(r"\s+", " ", q)[:40]
        if not q or self._has_foreign_word(q) or any(p in q for p in self._banned):
            return None
        return q

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
