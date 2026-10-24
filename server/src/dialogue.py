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

from .config import DialogueConfig, ProcedureConfig, config
from .knowledge import JST, Reference, SearchResult, Weather
from .persona import (
    Persona,
    build_system_prompt,
    build_turn_message,
    format_today,
    organization_word,
    source_labels,
    this_weekend,
)
from .procedure_kb import PROCEDURE_TRUST, ProcedureRecord
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

# 逆接で前置きを切る読点（「〜ですが、」「〜けど、」）
_CONCESSIVE_RE = re.compile(r"(が|けど|けれど|けれども|ものの)、")

# 話者の記号（WP02 B）。LLM には「A：〜」で渡す。応答の頭に付けてきたら外す
_SPEAKER_LABEL_RE = re.compile(r"^[A-Z]{1,2}[：:]\s*")


def grounding_overlap(sentence: str, source: str) -> float:
    """文の文字 2-gram のうち、資料にもあるものの割合（句読点・空白は除く）。"""
    def grams(t: str) -> set[str]:
        t = _ECHO_IGNORE_RE.sub("", t)
        return {t[i:i + 2] for i in range(len(t) - 1)}

    g = grams(sentence)
    return len(g & grams(source)) / len(g) if g else 0.0


def labeled(text: str, speaker: str | None) -> str:
    """LLM と履歴に渡す来訪者の発話（話者の記号があれば「A：〜」）。"""
    return f"{speaker}：{text}" if speaker else text


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
_DATE_RE = re.compile(r"[0-9０-９]{1,2}月[0-9０-９]{1,2}日")


def _short_name(name: str) -> str:
    """相談先の名前の括弧書き（説明）を除いた呼び名。"""
    return re.split(r"[（(]", name, maxsplit=1)[0].strip()


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
        procedure_cfg: ProcedureConfig | None = None,
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
        self._proc_cfg = procedure_cfg or config.procedure
        self._draft_covered_re = re.compile(self._proc_cfg.draft_covered_pattern)
        # このターンで定型に切り替えた安全装置（crisis / lost_child / ai / privacy。WP09 C）。無ければ None
        self.last_safety: str | None = None
        c = self._cfg
        self._safety = [(k, re.compile(p)) for k, p in (
            ("crisis", c.crisis_pattern), ("lost_child", c.lost_child_pattern),
            ("other_person", c.other_person_pattern),
            *((f"cannot_{k}", p) for k, p in c.cannot_patterns),
            ("opinion", c.opinion_pattern), ("ai", c.ai_question_pattern),
            ("privacy", c.privacy_question_pattern), ("complaint", c.complaint_pattern))]
        # このターンで当たった手続きナレッジ（承認済み・下書き。文字記録用）
        self.last_procedures: list[ProcedureRecord] = []
        # この会話で「詳しいことは〇〇課でご案内します」を足した手続き（同じ手続きで繰り返さない）
        self._draft_noted: set[str] = set()
        # 直近の応答が出した行為の提案（<propose:…>）。無ければ None
        self.last_proposal: str | None = None
        self._now = now
        self._ascii_re = re.compile(rf"[A-Za-zＡ-Ｚａ-ｚ]{{{self._cfg.ascii_run_min},}}")
        self._allow_ascii = {a.lower() for a in (
            (facility.allow_ascii if facility else ()) + self._cfg.allow_ascii_common)}
        self._banned = tuple(self._cfg.banned_phrases) + (facility.outside_places if facility else ())
        self._llm_source_re = re.compile(self._cfg.llm_source_sentence_pattern)
        self._mentions_source_re = re.compile(r"(?:ホームページ|サイト|お知らせ).{0,8}(?:載って|ご覧)")
        self._known_facilities = tuple(
            f.name for f in (facility.facilities_in_building if facility else ()))
        self._location_re = re.compile(self._cfg.building_location_pattern)
        self._facility_question_re = re.compile(self._cfg.facility_question_pattern)
        # 直近のターンの検索結果（計測・出典表示用）
        self.last_search_ms: float | None = None
        self.last_references: list[Reference] = []
        self.last_cited: list[Reference] = []
        # 直近のターンの検索語（日付を足したもの）と、安全装置で読み上げなかった文（文字記録・誤り探し用）
        self.last_search_query: str | None = None
        self.last_dropped: list[dict[str, str]] = []
        self._building_claim_res = [re.compile(p) for p in self._cfg.building_claim_patterns]
        self._building_context_re = re.compile(self._cfg.building_context_pattern)
        self._smalltalk_re = re.compile(self._cfg.smalltalk_pattern)
        self._medical_judgment_re = re.compile(
            rf"(?:{self._cfg.medical_subject_pattern})[^。！？!?]{{0,20}}(?:{self._cfg.judgment_pattern})")
        self._legal_judgment_re = re.compile(self._cfg.legal_judgment_pattern)
        self._meta_leak_re = re.compile(self._cfg.meta_leak_pattern)
        self._money_deadline_re = re.compile(self._cfg.money_deadline_pattern)
        self._bring_claim_re = re.compile(self._cfg.bring_claim_pattern)
        self._bring_item_re = re.compile(self._cfg.bring_item_pattern)
        self._future_re = re.compile(self._cfg.future_word_pattern)
        self._claim_re = re.compile(self._cfg.grounding_claim_pattern)
        self._dangling_re = re.compile(self._cfg.grounding_dangling_pattern)
        # 期限・金額の根拠にしてよい固定事実（開庁時間・補足・緊急連絡先）
        self._fact_texts = ([facility.hours, *facility.notes, *(f"{e.name}{e.number}" for e in facility.emergency_lines)]
                            if facility else [])
        self._residence_question_re = re.compile(self._cfg.residence_question_pattern)
        self._residence_allow_re = re.compile(self._cfg.residence_question_allow_pattern)
        # (user, assistant) の組を直近 history_turns 件だけ保持する
        self._history: deque[tuple[str, str]] = deque(maxlen=self._cfg.history_turns)
        self._question_re = re.compile(self._cfg.question_sentence_pattern)
        self._personal_question_re = re.compile(self._cfg.personal_question_pattern)
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
        self._draft_noted.clear()
        logger.info("対話履歴をクリアしました")

    def build_messages(
        self, text: str, language: str, refs: list[Reference] | None = None, hint: str = "",
        speaker: str | None = None,
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
        said = labeled(text, speaker)
        content = said if language == "ja" else f"[language: {language}] {said}"
        # 履歴には発話だけを積む（参考資料は今のターンにだけ付ける）
        # hint は今のターンにだけ添える材料（さっき来た方の小ネタなど）
        messages.append({"role": "user",
                         "content": hint + build_turn_message(self._cfg, self._facility, refs or [], content,
                                                              self._proc_cfg.section_head)})
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

    def begin_turn(self) -> None:
        """ターンごとの記録（提案・検索・出典・捨てた文）を消す。LLM を通さない定型文のターンでも前のものを残さない。"""
        self.last_proposal = None
        self.last_safety = None
        self.last_procedures = []
        self.last_dropped = []
        self.last_search_query = None
        self.last_search_ms = None
        self.last_references, self.last_cited = [], []

    def replace_last_reply(self, user: str, old: str, new: str) -> None:
        """直前のターン (user, old) の応答を、実際に読み上げた文 new に置き換える（提案のために落とした問いを残さない）。

        定型の聞き返しで答えたときなどはターンを積んでいないので、直前が (user, old) でなければ何もしない。
        """
        if self._history and self._history[-1] == (user, old):
            self._history[-1] = (user, new)

    def is_question(self, sentence: str) -> bool:
        """問いの文か（「〜ですか？」で終わる文など。dialogue.question_sentence_pattern）。"""
        return self._question_re.search(sentence.strip()) is not None

    def add_turn(self, user: str, assistant: str) -> None:
        """LLM を通さずに答えたターン（行為の定型文など）を履歴に積む。"""
        self._push(user, assistant)

    def _push(self, user: str, assistant: str) -> None:
        self._history.append((user, assistant))

    def restore(self, snapshot: list[tuple[str, str]], user: str, assistant: str) -> None:
        """履歴を snapshot（history で取ったもの）に戻し、1 ターンを足す（紹介状の聞き取りを 1 行にまとめる）。

        聞き取りのターンが多いと maxlen で前の話が押し出されるので、位置ではなく写しで戻す。
        """
        self._history.clear()
        self._history.extend(snapshot)
        self._push(user, assistant)

    async def respond(
        self,
        text: str,
        language: str = "ja",
        extra_refs: list[Reference] | None = None,
        hint: str = "",
        speaker: str | None = None,
    ) -> AsyncIterator[str]:
        """1 発話に対する応答を 1 文ずつ返す。

        Args:
            text: 来訪者の発話（ASR の確定テキスト、または text モードの入力）
            language: 検出言語（ISO 639-1）
            speaker: 話者の記号（WP02 B。「A」「B」…）。LLM には「A：〜」の形で渡し、履歴にもその形で積む

        Yields:
            応答の文。LLM が 1 文も返せなかった場合は定型の聞き返し文を 1 つ。
            履歴には、言い終えた全文を 1 ターンとして積む。
        """
        self.begin_turn()
        text = text.strip()
        if not text:
            return

        fixed = self.safety_reply(text)
        if fixed is not None:
            self._push(labeled(text, speaker), "".join(fixed))
            for line in fixed:
                yield line
            return

        if language != "ja":
            # 多言語は後日。LLM に任せると日本語で中身を答えてしまうので、決まった文を返す
            logger.info("日本語以外の発話（%s）には準備中の旨を返します", language)
            yield self._cfg.foreign_reply
            return

        asked = self._asked_unknown_facility(text)
        if asked is not None:
            # 登録の無い館内の設備の場所・有無は、場所を作らせないよう決まった文で答える
            reply = self._cfg.unknown_facility_sentence.format(name=asked)
            logger.info("登録の無い館内の設備（%s）を聞かれたので、決まった文で答えます", asked)
            self._push(labeled(text, speaker), reply)
            yield reply
            return

        refs = list(extra_refs) if extra_refs is not None else await self._search(text)
        if extra_refs is not None:
            self.last_references, self.last_search_ms = refs, None
        # 手続きナレッジ（WP09）: 承認済みのものだけ【市の案内（承認済み）】として参考資料より前に置く。
        # 下書きだけに当たったら、資料には入れず、最後に「詳しいことは〇〇課でご案内します」を足す
        proc_refs, draft_dept = self._procedure_refs(text) if extra_refs is None else ([], None)
        if proc_refs:
            refs = proc_refs + refs
            self.last_references = refs
        place = self._unknown_place(text, refs)
        if place is not None:
            # 地区・町の様子を聞かれたのに資料が無い: 様子を作らず、教えてもらう（人物像の「土地勘がまだ薄い」）
            reply = self._cfg.unknown_place_reply.format(place=place)
            logger.info("資料の無い地区の様子を聞かれたので、教えてもらう文で返します")
            self._push(labeled(text, speaker), reply)
            yield reply
            return
        messages = self.build_messages(text, language, refs, hint, speaker)
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
        cautioned = False
        bring_cautioned = False
        # 下書きの手続きに当たったのに規則で課が決まらない: LLM が言う課は推測なので読み上げない（WP09 C の評価で
        # 移住・体育館・ふるさと納税を社会福祉課と言った）
        guess_guard = draft_dept is not None and expected_dept is None
        allowed_depts = {draft_dept} if guess_guard else set()
        # 読点で早めに切った 1 文目の頭（「納期限は、」）が期限・金額に触れていたら、続きと合わせてから根拠を見る
        # （先に読み上げると、続きを捨てたときに「納期限は、詳しい期限や金額は…」とつながってしまう）
        held = ""
        try:
            async for delta in self._stream(messages):
                for raw in splitter.feed(think.feed(delta)):
                    sentence = held + await self._finish_sentence(raw, cited)
                    held = ""
                    if sentence.endswith(("、", ",")) and self._money_deadline_re.search(sentence):
                        held = sentence
                        continue
                    fixed_name = self._fix_unknown_department(sentence, expected_dept or (
                        draft_dept if draft_dept != self._proc_cfg.draft_department_fallback else None))
                    if fixed_name is None:
                        continue
                    sentence, dept_state = self._fix_department(fixed_name, expected_dept, dept_state)
                    sentence = self._fix_floor_of_last_department(sentence, sentences)
                    if guess_guard and self._names_department(sentence, allowed_depts):
                        # 規則で課が決まらない手続き（下書き）で LLM が課を言った: 推測なので言わない（最後に「担当の課で」）
                        logger.info("規則で課が決まらない手続きで、LLM が言った課の文を読み上げずに捨てました")
                        self.note_drop("規則で決まらない課の推測", sentence)
                        continue
                    if self._stale_future(sentence):
                        continue
                    if self._ungrounded_bring(sentence, refs):
                        if bring_cautioned:
                            continue
                        sentence, bring_cautioned = self._cfg.bring_caution, True
                    elif self._ungrounded_claim(sentence, refs):
                        # 根拠の無い期限・金額・支払方法: 捨てて、確認をすすめる一言を 1 回だけ
                        if cautioned:
                            continue
                        sentence, cautioned = self._cfg.grounding_caution, True
                    elif cautioned and self._dangling_re.search(sentence):
                        logger.info("捨てた期限・金額を指す文を読み上げずに捨てました")
                        self.note_drop("捨てた期限・金額を指す文", sentence)
                        continue
                    sentence = self._strip_judgment_clause(sentence)
                    if self._must_drop(sentence) or self._drop(sentence, text, sentences, previous):
                        dropped_echo = dropped_echo or not sentences
                        continue
                    if dropped_echo and not sentences:
                        # おうむ返しの断片を捨てた後の続き（「は、保健医療課で…」）の頭の助詞を外す
                        sentence = _LEADING_PARTICLE_RE.sub("", sentence)
                    if sentence:
                        sentence = self._attribute(sentence, proc_refs, sentences, cited)
                        sentences.append(sentence)
                        yield sentence
        except Exception:
            logger.exception("LLM 呼び出しに失敗しました")
            if not sentences:
                yield _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])
                return
            # 言いかけの残りは捨て、言えた文だけを履歴に積む
        else:
            tail = held + await self._finish_sentence(splitter.flush(), cited)
            tail = self._fix_unknown_department(tail, expected_dept or (
                draft_dept if draft_dept != self._proc_cfg.draft_department_fallback else None)) or ""
            tail, dept_state = self._fix_department(tail, expected_dept, dept_state)
            tail = self._fix_floor_of_last_department(tail, sentences)
            if tail and guess_guard and self._names_department(tail, allowed_depts):
                self.note_drop("規則で決まらない課の推測", tail)
                tail = ""
            if tail and self._stale_future(tail):
                tail = ""
            if self._ungrounded_bring(tail, refs):
                tail = "" if bring_cautioned else self._cfg.bring_caution
            elif self._ungrounded_claim(tail, refs):
                tail = "" if cautioned else self._cfg.grounding_caution
            elif cautioned and self._dangling_re.search(tail):
                self.note_drop("捨てた期限・金額を指す文", tail)
                tail = ""
            if tail and dropped_echo and not sentences:
                tail = _LEADING_PARTICLE_RE.sub("", tail)
            # 何も言っていないときの最後の文は捨てない（応答が空になる）
            tail = self._strip_judgment_clause(tail)
            if tail and not self._must_drop(tail) and (
                    not sentences or not self._drop(tail, text, sentences, previous)):
                tail = self._attribute(tail, proc_refs, sentences, cited)
                sentences.append(tail)
                yield tail
        self.last_cited = [refs[i - 1] for i in sorted(cited) if 0 < i <= len(refs)]
        already = any(self._mentions_source_re.search(x) for x in sentences)
        # 出典の一言は、応答が資料の中身を実際に使ったときだけ（雑談に関係の薄い資料の番号が付いても足さない）。
        # 手続きナレッジは「市の案内では、」を添えたので、ホームページの一言には数えない
        page_cited = [r for r in self.last_cited if r.trust != PROCEDURE_TRUST]
        if page_cited and not already and draft_dept is None and self._uses_references(sentences, page_cited):
            # 出典の一言は、使った資料の出どころから決めて足す（「観光協会のホームページ」など）。
            # 応答がすでに「〇〇のホームページに載っています」と言っていれば重ねない
            line = self._cfg.source_sentence.format(
                labels=source_labels(self._cfg, self._facility, page_cited))
            sentences.append(line)
            yield line
        if draft_dept is not None and sentences and not self._points_to_department(sentences):
            line = self._proc_cfg.draft_sentence.format(department=draft_dept)
            logger.info("承認前の手続きなので、詳しいことは担当の課で、の一文を足します")
            sentences.append(line)
            yield line
        # 緊急連絡先の名前を言ったのに番号を言っていなければ、番号の文を足す（いちばん大事な事実なので）
        for line in self._missing_emergency_numbers("".join(sentences)):
            sentences.append(line)
            yield line

        if not sentences:
            # 来訪者が同じことを聞き直した: 前と同じ答えしか無いなら、それを言い直す（全部を繰り返しとして捨てると聞き返しになる）
            again = [d["text"] for d in self.last_dropped if d["reason"] == "同じ文の繰り返し"][:2]
            if again:
                logger.info("聞き直されたので、前と同じ答えを言い直します")
                for line in again:
                    sentences.append(line)
                    yield line
        if not sentences:
            yield _FALLBACK_REPLY.get(language, _FALLBACK_REPLY["ja"])
            return
        reply = "".join(sentences)
        self._push(labeled(text, speaker), reply)
        logger.debug("応答: %s", reply)

    def safety_reply(self, text: str) -> list[str] | None:
        """定型に切り替える発話なら、その文（LLM を通さない。WP09 C）。深刻な打ち明けが最優先。

        crisis（自死・暴力・虐待の打ち明け）と lost_child（保護者を探す子ども）は、職員を呼ぶ（last_safety を見た
        呼び出し側が職員パネルに知らせる）。診断も助言もしない。
        """
        kind = next((k for k, r in self._safety if r.search(text)), None)
        if kind is None:
            return None
        self.last_safety = kind
        cfg = self._cfg
        self.last_proposal = None
        if kind == "crisis":
            line, others = self._crisis_lines(text)
            lines = [x.format(consult=self._consult_sentence(), line=line, others=others) for x in cfg.crisis_lines]
        elif kind == "lost_child":
            lines = list(cfg.lost_child_lines)
        elif kind == "other_person":
            lines = list(cfg.other_person_lines)
        elif kind == "opinion":
            lines = list(cfg.opinion_lines)
            election = next((d.name for d in (self._facility.departments if self._facility else ())
                              if "選挙" in d.name), None)
            if election and re.search(r"選挙|投票|候補", text):
                lines.append(cfg.opinion_election_sentence.format(department=self._department_line(election)))
        elif kind == "complaint":
            lines = list(cfg.complaint_lines)
            self.last_proposal = "refer" if "refer" in self._enabled_actions else None
        elif kind.startswith("cannot_"):
            sub = kind.removeprefix("cannot_")
            dept = next((self._department_line(d.name) for d in (self._facility.departments if self._facility else ())
                         if not d.place and any(k in h for h in d.handles for k in cfg.money_consult_keywords)), None)
            lines = [x.format(department=dept) for x in cfg.cannot_lines.get(sub, ()) if dept or "{department}" not in x]
            if sub in ("apply", "meet"):
                self.last_proposal = "refer" if "refer" in self._enabled_actions else None
        elif kind == "ai":
            role = self._persona.role_title if self._persona else "案内役"
            lines = [x.format(role_title=role) for x in cfg.ai_lines]
        else:
            lines = [x.format(days=config.transcript.retention_days) for x in cfg.privacy_lines]
        if kind in ("crisis", "lost_child"):
            logger.warning("定型に切り替えました（%s）", kind)
        else:
            logger.info("定型で答えます（%s）", kind)
        return [x for x in lines if x]

    def crisis_topic(self, text: str) -> str:
        """深刻な打ち明けの種類（abuse / dv / kokoro / other）。"""
        return next((k for k, p in self._cfg.crisis_topics if re.search(p, text)), "other")

    def _crisis_lines(self, text: str) -> tuple[str, str]:
        """種類に合う相談先の番号を 1 つ（読み上げる文）と、残りの相談先（字幕だけの括弧書き）。"""
        lines = [e for e in (self._facility.emergency_lines if self._facility else ()) if e.crisis]
        if not lines:
            return "", ""
        topic = self.crisis_topic(text)
        pick = next((e for e in lines if e.crisis == topic), None) or next((e for e in lines if e.crisis == "other"), None)
        said = ""
        if pick is not None:
            said = self._cfg.crisis_line_sentence.format(name=_short_name(pick.name), number=pick.spoken or pick.number)
        rest = [f"{_short_name(e.name)} {e.number}" for e in lines if e is not pick]
        others = self._cfg.crisis_others_sentence.format(lines="／".join(rest)) if rest else ""
        return said, others

    def _consult_sentence(self) -> str:
        """深刻な打ち明けの相談先（facility のこの建物の課から。無ければ crisis_consult_fallback）。"""
        f = self._facility
        if f is None:
            return self._cfg.crisis_consult_fallback
        keys = self._cfg.crisis_consult_keywords
        places = [self._department_line(d.name) for d in f.departments if not d.place
                  and any(k in h for h in d.handles for k in keys)]
        if not places:
            return self._cfg.crisis_consult_fallback
        return self._cfg.crisis_consult_sentence.format(places="や".join(places))

    def _procedure_refs(self, text: str) -> tuple[list[Reference], str | None]:
        """発話で言った手続きナレッジ。承認済みは参考資料にし、下書きだけなら足す一文の課（WP09）。"""
        if self._procedures is None:
            return [], None
        matched = self._procedures.matched(text)
        self.last_procedures = matched
        if not matched:
            return [], None
        usable = [r for r in matched if self._procedures.book is not None and self._procedures.book.usable(r)]
        if usable:
            refs = [r.to_reference(self._department_line(r.department)) for r in usable[: self._proc_cfg.max_per_turn]]
            logger.info("承認済みの手続きナレッジを %d 件、市の案内として入れます", len(refs))
            return refs, None
        fresh = [r for r in matched if r.id not in self._draft_noted]
        if not fresh or not re.search(f"{self._cfg.visitor_question_pattern}|{config.referral.procedure_pattern}|"
                                      f"{self._proc_cfg.draft_request_pattern}", text):
            return [], None
        self._draft_noted.update(r.id for r in fresh)
        # 担当課: 別名表の規則 → 下書きの課（対象一覧の handles の規則か、出典のページの課。LLM が選んだ課ではない）が 1 つなら
        drafted = {r.department for r in matched if r.department}
        dept = (self._expected_department(text) or (next(iter(drafted)) if len(drafted) == 1 else None)
                or self._proc_cfg.draft_department_fallback)
        logger.info("承認前（下書き）の手続きに当たりました。手続きナレッジは使いません")
        return [], dept

    def _names_department(self, sentence: str, allowed: set[str | None]) -> bool:
        """allowed 以外の課の名前を言う文か。"""
        return any(d.name in sentence and d.name not in allowed
                   for d in (self._facility.departments if self._facility else ()))

    def _fix_unknown_department(self, sentence: str, expected: str | None) -> str | None:
        """拠点に無い課の名前（「住民課」「高齢者医療課」）を、規則で決まる課に直す。決まらなければ文を捨てる（None）。"""
        if self._facility is None or not self._facility.departments:
            return sentence  # 課の一覧が無い拠点では見ない
        names = [d.name for d in self._facility.departments]
        out = sentence
        for m in re.finditer(r"[一-龥ァ-ヶー]{1,8}課", sentence):
            word = m.group(0)
            if any(n in word or word in n for n in names) or word.endswith(("担当の課", "各課", "の課")):
                continue
            if word in ("担当課", "窓口課", "関係課", "所管課", "課") or any(word in n for n in self._known_facilities):
                continue
            if expected is None:
                logger.info("拠点に無い課の名前を言う文を読み上げずに捨てました")
                self.note_drop("拠点に無い課の名前", sentence)
                return None
            logger.info("拠点に無い課の名前を、規則で決まる課に直しました")
            out = out.replace(word, expected)
        return out

    def _stale_future(self, sentence: str) -> bool:
        """過ぎた日付（年つき）を、これからのこととして言う文か（読み上げない）。"""
        d = latest_dated(sentence)
        if d is None or d >= self._now().date() or not self._future_re.search(sentence):
            return False
        logger.info("過ぎた日付をこれからのこととして言う文を読み上げずに捨てました")
        self.note_drop("過ぎた日付をこれからと言う", sentence)
        return True

    def _ungrounded_bring(self, sentence: str, refs: list[Reference]) -> bool:
        """持ち物・必要書類を言う文で、書類の名前が参考資料と固定事実のどれにも無いか（WP09）。"""
        if not sentence or self.is_question(sentence) or not self._bring_claim_re.search(sentence):
            return False
        items = [m.group(0) for m in self._bring_item_re.finditer(sentence)]
        if not items:
            return False
        sources = "".join(f"{r.title}{r.text}" for r in refs) + "".join(t for t in self._fact_texts if t)
        sources = _ECHO_IGNORE_RE.sub("", sources)
        missing = [i for i in items if _ECHO_IGNORE_RE.sub("", i) not in sources
                   and not any(_ECHO_IGNORE_RE.sub("", i)[-k:] in sources and len(i) - k <= 2 for k in range(2, len(i)))]
        if not missing:
            return False
        logger.info("参考資料に無い持ち物・書類を言う文を読み上げずに捨てました（%d 個）", len(missing))
        self.note_drop("根拠の無い持ち物・書類", sentence)
        return True

    def _points_to_department(self, sentences: list[str]) -> bool:
        """応答がもう担当の課での案内を言っているか（課の名前か「担当の課」と、案内の言い方が同じ文にある）。"""
        names = [d.name for d in (self._facility.departments if self._facility else ())] + ["担当の課", "窓口"]
        return any(n in x for x in sentences for n in names) or any(self._cfg.grounding_caution in x for x in sentences)

    def _department_line(self, name: str) -> str:
        """手続きナレッジの担当の書き方（「保健医療課（2階）」。階は facility から引く）。"""
        d = next((d for d in (self._facility.departments if self._facility else ()) if d.name == name), None)
        if d is None:
            return name
        if d.floor:
            return f"{name}（{d.floor}）"
        return f"{name}（この建物ではなく {d.place}）" if d.place else name

    def _attribute(self, sentence: str, proc_refs: list[Reference], said: list[str], cited: set[int]) -> str:
        """手続きナレッジから答えた最初の文に「市の案内では、」を一度だけ添える（WP09）。"""
        if not proc_refs or not sentence:
            return sentence
        prefix = self._proc_cfg.attribution_prefix.format(org=organization_word(self._facility))
        if any(prefix.rstrip("、") in x for x in [*said, sentence]):
            return sentence
        for i, r in enumerate(proc_refs, 1):
            # 手続き名・担当の行は除く（課の案内は固定事実なので、市の案内とは言わない）
            lines = [x for x in r.text.splitlines() if not x.startswith(("手続き:", "担当:"))]
            body = "\n".join(lines)
            items = [re.sub(r"（[^）]*の場合）$", "", y).strip() for x in lines
                     for y in re.split(r"[、→]", x.split(": ", 1)[-1]) if len(y.strip()) >= 4]
            if grounding_overlap(sentence, body) >= self._proc_cfg.attribution_min_overlap \
                    or any(_ECHO_IGNORE_RE.sub("", y) in _ECHO_IGNORE_RE.sub("", sentence) for y in items):
                cited.add(i)
                m = re.match(r"(はい|ええ|そうですね|それでしたら|それなら)、", sentence)
                return sentence[:m.end()] + prefix + sentence[m.end():] if m else prefix + sentence
        return sentence

    def _rain_likely(self, day: str = "今日") -> bool:
        """day（今日・明日）の予報の降水確率に、傘をすすめる基準以上のものがあるか（天気が無ければ判断しない＝True）。

        明日だけ雨のときに、今日の話で傘をすすめない（WP07 の評価で、今日は晴れなのに「傘を持ち歩くと」と言った）。
        """
        if self._weather is None:
            return True
        every = self._weather.lines(self._now())
        lines = [ln for ln in every if ln.startswith(day)] or every  # その日の予報が無ければ全部で見る
        pops = [int(x) for line in lines for x in re.findall(r"(\d+)パーセント", line)]
        return not pops or max(pops) >= self._cfg.umbrella_min_pop

    def _expected_department(self, text: str) -> str | None:
        if self._procedures is None or self._facility is None:
            return None
        cands = self._procedures.departments_for(text, self._facility)
        return cands[0] if len(cands) == 1 else None

    def _fix_floor_of_last_department(self, sentence: str, said: list[str]) -> str:
        """課の名前の無い「1階ロビーの窓口にあります」を、この応答で直前に出た課の階に直す。"""
        every = {d.name: d.floor for d in (self._facility.departments if self._facility else ())}
        depts = {n: f for n, f in every.items() if f}
        claim = re.compile(r"[0-9０-９一二三四五]階(ロビー)?(の窓口)?(に|で)(あり|ござい)")
        if every and claim.search(sentence) and not any(n in sentence for n in every):
            # 前に課が出ていない、または直前の課がこの建物に無い（にこ♡にこハウスなど）のに「2階にあります」: 階の作り話
            last_any = next((n for x in reversed(said) for n in every if n in x), None)
            if last_any is None or not every[last_any]:
                logger.info("どの課か分からない・この建物に無い課の「N階にあります」を読み上げずに捨てました")
                self.note_drop("この建物に無い課・どの課か分からない階", sentence)
                return ""
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
            # 「社会福祉課（2階）か保健医療課（2階）」のように、担当でない課を選択肢に並べたら外す
            others = "|".join(re.escape(n) for n in said if n != expected)
            if others:
                fixed = re.sub(rf"(?:{others})(?:（[^）]*）)?(?:か|または|や|、)、?(?={re.escape(expected)})", "", sentence)
                if fixed != sentence:
                    logger.info("担当でない課を選択肢に並べた部分を外しました（%s）", expected)
                    return fixed, True
            return sentence, True
        wrong = said[0]
        logger.info("担当課の取り違えを直しました（%s → %s）", wrong, expected)
        fixed = sentence.replace(wrong, expected, 1)
        # 「市民課（1階）」の階、「子育て支援課（にこ♡にこハウス）」の場所も、直した課のものに直す
        fixed = self._fix_department_floor(fixed)
        return fixed, True

    def _fix_department_floor(self, sentence: str) -> str:
        """「1階の保健医療課」「税務課（2階）」「市民課は2階」の階を、施設の事実の階に直す。"""
        for d in (self._facility.departments if self._facility else ()):
            if not d.floor or d.name not in sentence:
                continue
            n, floor = re.escape(d.name), d.floor.replace("\\", "")
            fixed = re.sub(rf"[0-9０-９一二三四五]階(ロビー)?(の|にある)({n})", rf"{floor}\g<2>\g<3>", sentence)
            fixed = re.sub(rf"({n})(（|\(|は|が)[0-9０-９一二三四五]階", rf"\g<1>\g<2>{floor}", fixed)
            # 「担当は農業委員会事務局で、場所は4階です」（課が 1 つだけの文の、場所は N 階）
            fixed = re.sub(rf"({n}[^。]{{0,12}}場所は)[0-9０-９一二三四五]階", rf"\g<1>{floor}", fixed)
            if fixed != sentence:
                logger.info("課の階を施設の事実に直しました（%s）", d.name)
                sentence = fixed
        for d in (self._facility.departments if self._facility else ()):
            if not d.floor and not d.place and d.name in sentence:
                # 階の分からない課（安芸太田は課の階が拠点 YAML に無い）の「〇〇課は本庁舎の2階にあります」は作り話
                m = re.search(rf"{re.escape(d.name)}(は|が)[^。]{{0,8}}[0-9０-９一二三四五]階(に|で)(あり|ござい)[^、。]*[、。]?", sentence)
                if m:
                    logger.info("階の分からない課の階を言う部分を外しました（%s）", d.name)
                    self.note_drop("階の分からない課の階", m.group(0))
                    sentence = sentence.replace(m.group(0), "")
                continue
            if d.floor or not d.place or d.name not in sentence:
                continue
            # この建物に無い課の「（2階）」「（にこ♡にこハウス）」の作り話を、拠点 YAML の場所に直す
            n = re.escape(d.name)
            head = re.split(r"[（(]", d.place, maxsplit=1)[0]
            fixed = re.sub(rf"({n})[（(](?![^）)]*{re.escape(head)})[^）)]{{0,20}}[）)]",
                           rf"\g<1>（この建物ではなく {d.place}）", sentence)
            fixed = re.sub(rf"[0-9０-９一二三四五]階(ロビー)?(の|にある)({n})", r"\g<3>", fixed)
            fixed = re.sub(rf"({n})(は|が)[0-9０-９一二三四五]階に", r"\g<1>\g<2>", fixed)
            if fixed != sentence:
                logger.info("この建物に無い課の場所を施設の事実に直しました（%s）", d.name)
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
            self.note_drop("会話の途中の挨拶", sentence)
            return True
        if "傘" in sentence and not self._rain_likely("明日" if "明日" in sentence or "明日" in text else "今日"):
            logger.info("降水確率が低いのに傘をすすめる文を読み上げずに捨てました")
            self.note_drop("傘（降水確率が低い）", sentence)
            return True
        if self._contradicts_persona(sentence):
            logger.info("人物像と食い違う文（生まれ・育ちなど）を読み上げずに捨てました")
            self.note_drop("人物像と食い違う", sentence)
            return True
        if self._unasked_self_intro(sentence, text):
            logger.info("聞かれていないのに名乗る文を読み上げずに捨てました")
            self.note_drop("聞かれていない名乗り", sentence)
            return True
        if (not said or len(_ECHO_IGNORE_RE.sub("", sentence)) >= 8) and self._is_echo(sentence, text):
            logger.info("来訪者の言葉を繰り返す文を読み上げずに捨てました")
            self.note_drop("おうむ返し", sentence)
            return True
        if self._is_repeat(sentence, previous):
            logger.info("これまでの応答と同じ文を読み上げずに捨てました")
            self.note_drop("同じ文の繰り返し", sentence)
            return True
        if self._too_many_questions(sentence, said):
            logger.info("1 回の応答で 2 つ目以降の問いを読み上げずに捨てました")
            self.note_drop("2 つ目以降の問い", sentence)
            return True
        if self._extra_help_offer(sentence, text):
            logger.info("「お困りのことは」「お変わりありませんか」の類を重ねない（1 会話 1 回・尋ねられた直後は言わない）")
            self.note_drop("お困りのことは・お変わりの重ね", sentence)
            return True
        return False

    def _asks_personal(self, sentence: str) -> bool:
        """氏名・電話番号・住所を尋ねる文か。LLM の応答は紹介状の聞き取りの外なので、言わずに捨てる。

        聞き取り（同意の後）の質問は actions.py の定型文で言い、この対話を通らない。
        住まいは「お住まい」「住所」「地区」を含む問いを広く止める（景色・季節が主語の雑談の問いは残す）。
        """
        if not sentence:
            return False
        residence = (self.is_question(sentence) and self._residence_question_re.search(sentence) is not None
                     and not self._residence_allow_re.search(sentence))
        if not residence and not self._personal_question_re.search(sentence):
            return False
        logger.warning("紹介状の聞き取りの外で、氏名・電話番号・住所を尋ねる文を LLM が出したので読み上げずに捨てました")
        logger.debug("捨てた文: %s", sentence)
        self.note_drop("聞き取りの外の個人情報の質問", sentence)
        return True

    def _ungrounded_claim(self, sentence: str, refs: list[Reference]) -> bool:
        """期限・金額・支払方法を言う文で、参考資料（と固定事実）のどれとも文字 2-gram の重なりが足りないか（WP07）。"""
        # 日付（「10月24日がごみの日」）も同じ照合に掛ける（WP09 の評価で、資料に無い収集日を言った）
        if not sentence or not (self._money_deadline_re.search(sentence) or _DATE_RE.search(sentence)):
            return False
        now = self._now()
        today = format_today(now.date(), self._weather.lines(now) if self._weather is not None else [])
        sources = ([f"{r.title}{r.text}".translate(_ZEN_DIGITS) for r in refs] + [t for t in self._fact_texts if t]
                   + [today.translate(_ZEN_DIGITS)])
        best = max((grounding_overlap(sentence, src) for src in sources), default=0.0)
        # 数（単位つき）と支払方法の語は、資料のどれかにそのまま無ければ根拠なし（主語の語だけで重なりが足りてしまうため）
        claims = self._claim_re.findall(sentence.translate(_ZEN_DIGITS))
        missing = [c for c in claims if not any(c in src for src in sources)]
        if best >= self._cfg.grounding_min_overlap and not missing:
            return False
        logger.info("参考資料に根拠の無い期限・金額・支払方法の文を読み上げずに捨てました（重なり %.2f、資料に無い語 %d）",
                    best, len(missing))
        self.note_drop("根拠の無い期限・金額・支払方法", sentence)
        return True

    def _must_drop(self, sentence: str) -> bool:
        """応答が 1 文だけでも読み上げない文（聞き取りの外の個人情報の質問・医療や法律の判断）。"""
        return self._asks_personal(sentence) or self._judges_medical_or_legal(sentence)

    def _strip_judgment_clause(self, sentence: str) -> str:
        """「〜の症状にもなり得ますが、まずは医師に相談を」の、判断する前置きの節だけを外す（後ろの案内は残す）。"""
        m = _CONCESSIVE_RE.search(sentence)
        if m is None:
            return sentence
        head, rest = sentence[:m.end()], sentence[m.end():]
        if rest and (self._medical_judgment_re.search(head) or self._legal_judgment_re.search(head)) \
                and not (self._medical_judgment_re.search(rest) or self._legal_judgment_re.search(rest)):
            logger.info("病気・法律の判断をする前置きの節を外しました")
            self.note_drop("医療・法律の判断", head)
            return rest
        return sentence

    def _judges_medical_or_legal(self, sentence: str) -> bool:
        """病名・症状・障害・認知症などの見立てや、法的な可否を言う文か（「〜かもしれません」「違法です」）。"""
        if not sentence or not (self._medical_judgment_re.search(sentence) or self._legal_judgment_re.search(sentence)):
            return False
        logger.warning("病気・法律の判断をする文を LLM が出したので読み上げずに捨てました")
        logger.debug("捨てた文: %s", sentence)
        self.note_drop("医療・法律の判断", sentence)
        return True

    def fixed_text_problems(self, text: str, refs: list[Reference], stale_future: bool = True) -> list[str]:
        """LLM の応答の外で作った読み上げ文（夜に作るお知らせの要約。WP11）を、読み上げ前の安全装置に通す。

        応答と違って文を直さない（直すと記事と食い違うので、呼び出し側が保留にする）。引っかかった理由
        （note_drop と同じ言い方）を返す。空なら通った。stale_future=False なら「過ぎた日付をこれからと言う」は
        呼び出し側で見る（お知らせは年の無い日付と「〜から開始しています」を別の規則で見る）。
        """
        self.begin_turn()
        said: list[str] = []
        for raw in re.split(r"(?<=[。！？!?])", text):
            s = raw.strip()
            if not s:
                continue
            if self._meta_leak_re.search(s):
                self.note_drop("指示文の漏れ", s)
            if any(p in s for p in self._banned):
                self.note_drop("言わない言い回し", s)
            if self._has_foreign_word(s):
                self.note_drop("英単語", s)
            if self._fix_unknown_department(s, None) is not None and (
                    self._fix_department_floor(s) != s or self._fix_floor_of_last_department(s, said) != s):
                self.note_drop("課の階・場所の作り話", s)
            if self._invented_building_place(s) is not None or self._unknown_facility_location(s) is not None:
                self.note_drop("館内の設備・場所の作り話", s)
            if self._department_directions(s):
                self.note_drop("館内の道順", s)
            if stale_future:
                self._stale_future(s)
            self._ungrounded_claim(s, refs)
            self._ungrounded_bring(s, refs)
            self._must_drop(s)
            said.append(s)
        return list(dict.fromkeys(d["reason"] for d in self.last_dropped))

    def note_drop(self, reason: str, sentence: str) -> None:
        """安全装置で読み上げなかった文を、このターンの記録に残す（文字記録と誤り探しの素材に使う）。"""
        self.last_dropped.append({"reason": reason, "text": sentence})

    def _invented_building_place(self, sentence: str) -> str | None:
        """館内の設備・場所が「〜があります」「〜にあります」と言う文で、登録に無い名前ならその名前。

        登録は facility.facilities_in_building・departments・施設名。トイレの場所を言う文は先に
        _unknown_facility_location が決まった文に差し替えるので、ここに来るのはそれ以外の言い方。
        """
        if self._facility is None or not self._building_context_re.search(sentence):
            return None
        known = [self._facility.name, *self._known_facilities, *(d.name for d in self._facility.departments)]
        for pat in self._building_claim_res:
            for m in pat.finditer(sentence):
                noun = m.group("noun").strip("、 ")
                if not noun:
                    continue
                if any(k in noun or noun in k for k in known if k):
                    continue
                return noun
        return None

    def _too_many_questions(self, sentence: str, said: list[str]) -> bool:
        """1 回の応答の問い（「〜ですか？」で終わる文）が上限を超えるか。"""
        if not self.is_question(sentence):
            return False
        return sum(1 for x in said if self.is_question(x)) >= self._cfg.max_questions_per_reply

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
        self.last_search_query = query
        refs = result.references
        if self._smalltalk_re.search(text):
            # 雑談寄り（挨拶・天気・人物像への質問）は関係の薄い資料が付きやすいので、しきい値を上げる
            # （「日本語うまいですね」に防災ガイドブックが付いた）
            weak = [r for r in refs if r.vector_score < self._cfg.smalltalk_min_vector_score]
            if weak:
                logger.info("雑談寄りの発話なので、類似度の低い資料を %d 件外しました", len(weak))
                refs = [r for r in refs if r not in weak]
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
        sentence = _SPEAKER_LABEL_RE.sub("", sentence)  # 来訪者の話者の記号（「A：」）をまねた頭は外す
        if self._avatar_name:  # 「いどちゃん: こんばんは」のような話者名は外す
            # 「いどちゃん: …」「いどちゃんは、…」「いどちゃん物忘れが…」の頭の名前を外す（「いどちゃんです」は残す）
            name = re.escape(self._avatar_name)
            # 「いどちゃんの応答：…」の形も外す（WP06 の評価で出た）
            sentence = re.sub(rf"^{name}(?:の(?:応答|返答|返事|答え|発言))?\s*(?:[:：、]|[はも]、)\s*"
                              rf"|^{name}(?=[一-龥ァ-ヶ])", "", sentence)
        for old, new in self._cfg.leak_replacements:
            sentence = sentence.replace(old, new)
        for old, new in (self._facility.name_fixes if self._facility else ()):
            sentence = sentence.replace(old, new)  # 団体の正式名（「江田島観光協会」→「江田島市観光協会」）
        if self._meta_leak_re.search(sentence):
            logger.warning("指示文のような文（来訪者・以下は など）を LLM が出したので読み上げずに捨てました")
            self.note_drop("指示文の漏れ", sentence)
            return ""
        if self._llm_source_re.search(sentence):
            return ""  # 出典の一言はこちらで足す
        sentence = self._fix_department_floor(sentence)
        unknown = self._unknown_facility_location(sentence)
        if unknown is not None:
            logger.info("館内の設備（%s）の場所を作った文を差し替えます", unknown)
            logger.debug("差し替え前: %s", sentence)
            self.note_drop("登録の無い設備の場所", sentence)
            return self._cfg.unknown_facility_sentence.format(name=unknown)
        invented = self._invented_building_place(sentence)
        if invented is not None:
            logger.info("館内に登録の無い設備・場所（%s）があると言う文を読み上げずに捨てました", invented)
            self.note_drop("館内の設備・場所の作り話", sentence)
            return ""
        if self._department_directions(sentence):
            logger.info("館内の道順を作った文を読み上げずに捨てました")
            self.note_drop("館内の道順", sentence)
            return ""
        if any(p in sentence for p in self._banned):
            logger.warning("言わないことにしている言い回しを含む文を読み上げずに捨てました")
            logger.debug("捨てた文: %s", sentence)
            self.note_drop("言わない言い回し", sentence)
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
            self.note_drop("英字の言い直しに失敗", sentence)
            return ""
        fixed = _SOURCE_MARK_RE.sub("", fixed).strip()
        if not fixed or self._has_foreign_word(fixed):
            logger.warning("言い直しても英字が残ったため、この文は読み上げません")
            self.note_drop("英字が残った", sentence)
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
