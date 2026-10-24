"""同意を取ってから外に働きかける（WP04 A-2 / A-3 / B-1）。

1 会話の流れ（Conversation.respond）:

- LISTEN: 通常の対話。LLM の応答の最後に <propose:lookup|refer|share> があれば、行為ごとの
  定型文で提案して PROPOSED へ（提案の言葉は LLM に作らせない）
- PROPOSED: 来訪者の発話は通常の対話に流さず、返事だけを規則で判定する（同意 / 辞退 / 不明）。
  不明なら 1 回だけ聞き直し、それでも不明なら辞退扱い
- 同意 → 「少々お待ちくださいね」→ 実行（エッジは thinking）→ 結果を 1〜3 文で報告 → LISTEN
- 紹介状で電話番号を言っていたら、復唱して確かめる（CONFIRM_PHONE）。「090の1234の5678」と区切って
  言われ ASR が断片で返したときは、数字が 10〜11 桁にそろうまで続く発話を連結してから復唱する
  （待つのは断片ごとに referral.phone_wait_sec。そろわなければ 1 回だけ聞き直し、だめなら番号なしで進める）
- 紹介状を保存したら会話を仕切り直す: 聞き取りの質問と答えを対話履歴から外して「受付番号 ○ で○○課に
  伝達済み」の 1 行だけを残し、保存直後の数字だけ・1〜2 語だけの断片は無視する（ignores）

会話の区切り（待機タイムアウト）で、匿名の来訪記録を 1 行書く（end_session）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .asr_client import ngram_overlap
from .clarify import OTHER, Clarifier, Clarify, load_clarify_rules
from .config import ActionsConfig, NewsConfig, ReferralConfig, TidbitConfig
from .dialogue import ChatBackend, Dialogue
from .knowledge import JST
from .lookup import Lookup
from .news import NewsBook, spoken_date, tidy_summary
from .persona import organization_word
from .procedure_kb import ProcedureRecord
from .procedures import AsrFixes, IntakeSet, Procedures
from .referrals import (
    CATEGORIES,
    CONTACT_PREFS,
    PHONE_DIGITS_MAX,
    UNSPECIFIED_DEPARTMENT,
    URGENCIES,
    ReferralStore,
    TidbitStore,
    VisitLog,
    find_phone,
    format_phone,
    normalize_digits,
    phone_digits,
    phone_for_speech,
    pick_tags,
)
from .site_context import Facility
from .timeofday import greeting_for

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ReplyWords:
    strong_yes: list[str]
    yes_override: list[str]
    no: list[str]
    yes: list[str]
    leading_ack: list[str] = field(default_factory=list)
    no_after_ack: list[str] = field(default_factory=list)


def load_templates(path: Path) -> tuple[ReplyWords, dict[str, dict[str, str]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    r = data.get("reply") or {}
    words = ReplyWords(*([str(x) for x in r.get(k) or []]
                         for k in ("strong_consent", "consent_override", "decline", "consent",
                                   "leading_ack", "decline_after_ack")))
    actions = {str(k): {str(a): str(b) for a, b in (v or {}).items()} for k, v in (data.get("actions") or {}).items()}
    return words, actions


def classify_reply(text: str, words: ReplyWords) -> str:
    """返事の判定（"yes" / "no" / "unclear"）。LLM は使わない。

    はっきりした同意（お願い・頼む）→ 辞退の語を含む同意（いいですよ）→ 辞退 → 同意、の順に見る。
    「はい、大丈夫です」は辞退（よくある言い方）、「いや、お願いします」は同意。
    """
    t = text.strip()
    ack = next((a for a in sorted(words.leading_ack, key=len, reverse=True) if t.startswith(a)), None)
    rest = re.sub(r"^[\s、。,.!！]+", "", t[len(ack):]) if ack else ""
    if rest:
        # 「うん、ないです」「はい、大丈夫です」: 頭の相槌より後ろの語を優先する
        if not any(w in rest for w in words.strong_yes + words.yes_override) and any(w in rest for w in words.no_after_ack):
            return "no"
        verdict = classify_reply(rest, words)
        if verdict != "unclear":
            return verdict
    if any(w in t for w in words.strong_yes) or any(w in t for w in words.yes_override):
        return "yes"
    if any(w in t for w in words.no):
        return "no"
    if any(w in t for w in words.yes):
        return "yes"
    return "unclear"


@dataclass
class Pending:
    """提案して返事を待っている行為。"""

    action: str
    user_text: str
    language: str
    # proposed / intake / ask_name / confirm_name / ask_area / ask_contact / ask_phone / confirm_phone / confirm_share
    phase: str = "proposed"
    reasks: int = 0
    share_text: str | None = None
    # 提案したターンの番号（紹介状の要約は、このきっかけの発話から後だけを使う）
    turn_index: int = 0
    # 提案の後の来訪者の返事のうち、話題のもの（同意と手続きの聞き取りの答え。要約に使う）
    replies: list[str] = field(default_factory=list)
    # 紹介状の聞き取り（同意の後）
    department: str = ""
    intake_set: IntakeSet | None = None
    intake: list[dict[str, str]] = field(default_factory=list)  # {question, ask, answer}
    intake_idx: int = 0
    name: str | None = None
    name_retry: bool = False
    address_area: str | None = None
    contact_pref: str = "未定"
    phone: str | None = None
    # 区切って言われた電話番号の断片を連結している途中の数字と、続きを待つ期限（monotonic 秒）
    phone_buf: str = ""
    phone_deadline: float | None = None
    phone_reasked: bool = False
    # 提案した時点の対話履歴の写し（保存後、聞き取りを外してここに戻す）
    history_before: list[tuple[str, str]] = field(default_factory=list)
    # 話者の記号（WP02 B）: 提案のきっかけを言った人と、同意を言った人（聞き取りの主。ほかの人の答えも受け付ける）
    speaker: str | None = None
    consent_speaker: str | None = None
    # 提案の返事待ちに黙られて、提案をもう一度言ったか（1 回だけ。WP02 C）
    silence_reasked: bool = False
    # 保存の後に、来訪者が触れた項目だけ聞き取りを再開したとき: 書き直す紹介状の (ID, 日付)（WP07）
    reopen: tuple[str, str] | None = None
    # 用件の聞き分けの問い（action が clarify のとき。WP13 E）
    clarify: Clarify | None = None
    # 聞き分けで決まらなかったときの紹介状: 担当課は未特定のまま（規則でも決めない）
    unspecified: bool = False


@dataclass
class Session:
    started_at: str
    # 文字記録で会話をまとめる ID（来訪記録には書かない）
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    language: str = "ja"
    turns: int = 0
    backchannel_count: int = 0
    proposals: list[dict[str, str | None]] = field(default_factory=list)
    referral_id: str | None = None
    # 小ネタ: この会話で紹介したか（1 会話 1 つまで）、この会話で預かったもの（本人には返さない）
    tidbit_used: bool = False
    own_tidbits: list[str] = field(default_factory=list)
    # お知らせ（WP11）: 提案した回数、この会話では提案しない理由（深刻な打ち明け・急いでいる）、続いた雑談のターン
    news_proposed: int = 0
    news_blocked: str | None = None
    smalltalk_streak: int = 0
    # 話題の手続き（手続きナレッジの id か別名表の語）と、それが続いたターン数、承認済みか、
    # その間に LLM が紹介状を提案しなかったターン数（WP11 の実機確認で足した「用件が一段落」の規則）
    topic: str | None = None
    topic_turns: int = 0
    topic_approved: bool = False
    topic_no_refer: int = 0
    # 用件の聞き分け（WP13 E）: この会話で聞いた候補の組（1 つの用件につき 1 回）
    clarified: set[frozenset[str]] = field(default_factory=set)


StateHook = Callable[[str], Awaitable[None]]
# 構図を変える通知（WP11。お知らせの layout メッセージ）
LayoutHook = Callable[[dict[str, Any]], Awaitable[None]]


class Conversation:
    """対話（Dialogue）に、相槌の数え上げ・行為の提案と実行・来訪記録を足したもの。"""

    def __init__(
        self,
        dialogue: Dialogue,
        llm: ChatBackend,
        model: str,
        cfg: ActionsConfig,
        referral_cfg: ReferralConfig,
        facility: Facility | None,
        tags: list[str],
        lookup: Lookup | None = None,
        referrals: ReferralStore | None = None,
        visits: VisitLog | None = None,
        on_state: StateHook | None = None,
        edge_id: Callable[[], str] = lambda: "",
        tidbits: TidbitStore | None = None,
        tidbit_cfg: TidbitConfig | None = None,
        procedures: Procedures | None = None,
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] = time.monotonic,
        asr_fixes: AsrFixes | None = None,
        speaker_count: Callable[[], int] | None = None,
        news: NewsBook | None = None,
        news_cfg: NewsConfig | None = None,
        on_layout: LayoutHook | None = None,
    ) -> None:
        self.dialogue = dialogue
        self._llm = llm
        self._model = model
        self._cfg = cfg
        self._ref_cfg = referral_cfg
        self._facility = facility
        self._tags = tags
        self._lookup = lookup
        self._referrals = referrals
        self._visits = visits
        self._on_state = on_state
        self._edge_id = edge_id
        self._tidbits = tidbits
        self._tidbit_cfg = tidbit_cfg or TidbitConfig()
        self._procedures = procedures or Procedures()
        self._asr_fixes = asr_fixes
        # この会話で相手として受け付けた話者の数（来訪記録に書く。WP02 B）と、いま話している人の記号
        self._speaker_count = speaker_count
        self._speaker: str | None = None
        # この会話のターン（来訪者, 応答）（プロセス内のみ）
        self._turns: list[tuple[str, str]] = []
        # 聞き取った氏名・地区・電話番号（画面・コンソールで伏せるため。会話の区切りで消す）
        self._private_values: set[str] = set()
        self._now = now or (lambda: datetime.now(JST))
        self._clock = clock
        # 紹介状を保存した時刻（monotonic。直後の断片を無視する）と、履歴に残す 1 行（保存の直後だけ）
        self._saved_at: float | None = None
        self._history_note: tuple[str, str] | None = None
        self._saving = False
        # この会話で保存した紹介状 (ID, 日付, 受付番号)。保存の後に来訪者が触れた項目を書き足すのに使う（WP07）
        self._saved_referral: tuple[str, str, str] | None = None
        # このターンの発話を、電話番号の続きを待つために返事をせずに預かったか
        self._held = False
        # このターンの行為（文字記録用）: 提案した行為、返事をした相手の行為と段、その結果
        self.turn_info: dict[str, str | None] = {}
        self._words, self._templates = load_templates(cfg.templates_path)
        # 用件の聞き分け（WP13 E）と、聞き分けで決まった手続き（紹介状の課・聞き取りに使う。会話の区切りで消す）
        self._clarifier = Clarifier(self._procedures, load_clarify_rules(cfg.templates_path), dialogue._proc_cfg)
        self._clarified: ProcedureRecord | None = None
        self._pending: Pending | None = None
        self._session: Session | None = None
        # 直前のターンで行為が終わったか（次の短い相づちを決まった文で受ける）
        self._after_action = False
        # 直近の紹介状の受付番号（字幕・表示用）
        self.last_receipt: str | None = None
        # お知らせ（WP11）: 記事、設定、構図を変える通知（進行役＝main.py が役を選び、エッジへ送る）、
        # 「市」「町」、このターンで用件が済んだか、紹介した後に main が言う締めの文と履歴の 1 行
        self._news = news
        self._news_cfg = news_cfg or NewsConfig(enabled=False)
        self._on_layout = on_layout
        self._org = organization_word(facility)
        self._business_done = False
        self._news_closing: str | None = None
        self._news_history: str | None = None
        self._hurry_re = re.compile(self._news_cfg.hurry_pattern)
        self._nothing_else_re = re.compile(self._news_cfg.nothing_else_pattern)
        self._anything_else_re = re.compile(self._news_cfg.anything_else_pattern)
        self._smalltalk_re = re.compile(self.dialogue._cfg.smalltalk_pattern)
        self._visitor_question_re = re.compile(self.dialogue._cfg.visitor_question_pattern)
        news_t = self._templates.get("news", {})
        self._news_ask_re = re.compile(news_t.get("ask_pattern") or r"(?!)")
        self._news_ask_request_re = re.compile(news_t.get("ask_request_pattern") or r"(?!)")
        self._closing_reply_re = re.compile(news_t.get("closing_reply_pattern") or r"(?!)")

    # ------------------------------------------------------------------
    # 会話の区切り
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        return "listen" if self._pending is None else self._pending.phase

    @property
    def session_id(self) -> str:
        """いまの会話の ID（まだ始まっていなければ始める。文字記録用）。"""
        self._ensure_session("ja")
        assert self._session is not None
        return self._session.id

    @property
    def session_language(self) -> str | None:
        """会話が始まっていればその言語（最初の発話の言語）。始まっていなければ None。"""
        return self._session.language if self._session is not None and self._session.turns else None

    @property
    def language(self) -> str:
        """返事待ちの行為の言語（無ければ会話の言語、まだ無ければ ja）。"""
        if self._pending is not None:
            return self._pending.language
        return self._session.language if self._session is not None else "ja"

    def silence_reask(self) -> str | None:
        """提案の返事待ちのまま黙られたら、提案の文（「〜しましょうか？」）をもう一度。1 回だけ（無ければ None）。"""
        p = self._pending
        if p is None or p.phase != "proposed" or p.silence_reasked:
            return None
        p.silence_reasked = True
        line = self._t(p.action, "propose")
        self.dialogue.extend_last_reply(line)
        logger.info("提案の返事待ちに黙っているので、提案をもう一度言います（%s）", p.action)
        return line

    def note_assistant_line(self, text: str) -> None:
        """案内役から言った定型の一言（沈黙・別れ）を履歴に積む（LLM が同じ問いを繰り返さないように。WP07）。

        直前の応答の後ろに足す（来訪者はその間に何も言っていない）。履歴が空なら切り出しと同じ形で 1 ターンにする。
        """
        if self.dialogue.history:
            self.dialogue.extend_last_reply(text)
        else:
            self.note_opener(text)

    def note_opener(self, text: str) -> None:
        """案内役から話しかけた切り出しを履歴に積む（LLM が挨拶を重ねない・同じ問いを繰り返さないように。WP02 C）。"""
        self._ensure_session("ja")
        self.dialogue.add_turn(self._cfg.opener_user_placeholder, text)

    def note_backchannel(self) -> None:
        # 相槌は応答より先に出るので、会話の最初の相槌のときはまだ会話が始まっていない
        self._ensure_session("ja")
        assert self._session is not None
        self._session.backchannel_count += 1

    def _ensure_session(self, language: str) -> None:
        if self._session is None:
            self._session = Session(started_at=datetime.now(JST).isoformat(timespec="seconds"),
                                    language=language)

    async def end_session(self) -> None:
        """会話の区切り: 来訪記録を 1 行書いて、履歴と提案を消す。"""
        p = self._pending
        if p is not None and p.action == "refer" and p.phase != "proposed" and self._referrals is not None:
            # 聞き取りの途中で立ち去った: 伺えたところまでで紹介状を作る（答えなくても作る）
            logger.info("聞き取りの途中で会話が区切られたので、伺えたところまでで紹介状を作ります")
            async for _ in self._finish_referral(p):
                pass
        session, history = self._session, self.dialogue.history
        self._session, self._pending = None, None
        self._saved_at, self._history_note = None, None
        self._saved_referral = None
        self._clarified = None
        self._news_closing, self._news_history, self._business_done = None, None, False
        self._turns = []
        self._private_values = set()
        self.dialogue.reset()
        if session is None or self._visits is None or not session.turns:
            return
        tags = await self._tag_conversation(history)
        record = {
            "started_at": session.started_at,
            "ended_at": datetime.now(JST).isoformat(timespec="seconds"),
            "turns": session.turns, "language": session.language, "tags": tags,
            "backchannel_count": session.backchannel_count, "proposals": session.proposals,
            "referral_id": session.referral_id, "edge_id": self._edge_id(),
            "speakers": self._speaker_count() if self._speaker_count is not None else None,
        }
        try:
            self._visits.write(record)
        except OSError:
            logger.warning("来訪記録を書けません", exc_info=True)

    # ------------------------------------------------------------------
    # 1 ターン
    # ------------------------------------------------------------------

    async def respond(self, text: str, language: str = "ja", speaker: str | None = None) -> AsyncIterator[str]:
        """1 発話に応答する。ターンを記録する。speaker は話者の記号（WP02 B。無ければ区別しない）。"""
        self._speaker = speaker
        self.dialogue.begin_turn()
        if self.ignores(text):
            logger.info("紹介状の保存直後の短い断片（数字だけ・1〜2 語）なので応答しません")
            return
        index = len(self._turns)
        label = self.private_label
        self._held = False
        p = self._pending
        self.turn_info = {"proposed": None, "reply_to": p.action if p else None,
                          "reply_phase": p.phase if p else None, "outcome": None}
        lines: list[str] = []
        async for s in self._respond_turn(text, language, index):
            lines.append(s)
            yield s
        logger.info("お知らせの判定: %s", self._news_status())
        if self._held and not lines:
            return  # 電話番号の断片: 続きを待つ（返事も記録もしない）
        # 氏名・地区・電話番号はプロセス内の記録にも残さない
        self._turns.append((f"（{label}を伺いました）" if label else text, self.redact("".join(lines))))

    async def _respond_turn(self, text: str, language: str, index: int) -> AsyncIterator[str]:
        self._ensure_session(language)
        assert self._session is not None
        if self._session.turns == 0:
            self._session.language = language
        self._session.turns += 1
        after_action, self._after_action = self._after_action, False
        self._business_done = False
        self._news_block_check(text)
        fixed = self.dialogue.safety_reply(text)
        # 不満・代わりの申請・市長に会わせての定型は、ご意見として伝える提案を付けるので、ふつうの流れ（dialogue.respond）で言う
        if fixed is not None and self.dialogue.last_safety in ("complaint", "cannot_apply", "cannot_meet") \
                and self._pending is None:
            fixed = None
        if fixed is not None and (self._pending is None or self.dialogue.last_safety in ("crisis", "lost_child")):
            # 深刻な打ち明け・迷子は、提案の返事待ち・紹介状の聞き取りの途中でも最優先で定型に切り替える
            if self._pending is not None:
                logger.info("返事待ち・聞き取りの途中ですが、安全装置の定型に切り替えます")
                self._outcome("declined")
                self._pending = None
            self.dialogue.add_turn(text, "".join(fixed))
            if self.dialogue.last_safety in ("crisis", "lost_child"):
                self._session.news_blocked = "深刻な打ち明け・迷子"  # この会話では以後お知らせを提案しない
            for line in fixed:
                yield line
            return
        self.dialogue.last_safety = None
        if self._is_news_ask(text) and (self._pending is None or self._pending.phase == "proposed"):
            # 来訪者が直接お知らせを聞いた: LLM に答えさせず、提案文を飛ばしてすぐ紹介する（返事待ちの提案は辞退として閉じる）
            if self._pending is not None:
                logger.info("提案の返事待ちにお知らせを聞かれたので、提案を辞退扱いにして紹介します（%s）", self._pending.action)
                self._outcome("declined")
                self._pending = None
            self._session.smalltalk_streak = 0
            async for s in self._present_news_direct(text, language, index):
                yield s
            return
        if (after_action and self._pending is None and classify_reply(text, self._words) != "unclear"
                and len(re.sub(r"[\s、。！？!?]", "", text)) <= self._cfg.reply_max_chars):
            # 行為が終わった直後の「うん」「ありがとう」は決まった文で受ける（お知らせを紹介できるなら、その提案）
            if self._news_available() and self._news_cfg.propose_after_business:
                line = self._t("news", "ack_prefix")
                yield line
                self.dialogue.add_turn(text, line)
                yield self._propose_news(text, language, index, "行為の後")
                return
            line = self._t("common", "ack")
            yield line
            self.dialogue.add_turn(text, line)
            return
        if self._pending is not None and self._pending.action == "clarify":
            async for s in self._on_clarify_reply(text, language, index):
                yield s
            return
        if self._pending is not None and self._is_new_topic(text):
            # 提案には答えず別の話を始めた: 聞き直さずに辞退として、その話に普通に答える
            logger.info("返事待ちに別の話が来たので、提案を辞退扱いにします")
            self._outcome("declined")
            self._pending = None
        if self._pending is not None:
            self._session.smalltalk_streak = 0
            if self._session.topic is not None:
                self._session.topic_turns += 1  # 提案への返事・聞き取りの答えも、手続きの話の続きに数える
            async for s in self._on_reply(text):
                yield s
            self._after_action = self._pending is None
            if self._business_done and self._news_available() and self._news_cfg.propose_after_business:
                # 紹介状の保存と持ち物の案内の後: お知らせを一つ紹介してよいか尋ねる
                yield self._propose_news(text, language, index, "紹介状の保存の後")
                self._after_action = False
            return
        if (self._news_cfg.propose_after_business and self._turns
                and self._anything_else_re.search(self._turns[-1][1]) and self._nothing_else_re.search(text.strip())
                and self._news_available()):
            # 「ほかに何かありますか」に「ないです」: 用件が済んだので、お知らせを一つ紹介してよいか尋ねる
            self._session.smalltalk_streak = 0
            self.dialogue.add_turn(text, "")
            yield self._propose_news(text, language, index, "ほかに無い")
            return
        s_ = self._session
        if (self._news_cfg.propose_after_business and s_.topic is not None
                and s_.topic_turns >= self._news_cfg.topic_turns and self._closing_reply_re.search(text.strip())
                and len(re.sub(r"[\s、。！？!?]", "", text)) <= self._cfg.reply_max_chars and self._news_available()):
            # 同じ手続きの話が続いた後の「わかりました」「ありがとう」: 用件が一段落したので、お知らせを一つ紹介してよいか尋ねる
            s_.smalltalk_streak = 0
            s_.topic, s_.topic_turns = None, 0
            self.dialogue.add_turn(text, "")
            yield self._propose_news(text, language, index, "手続きの話の後の締めの返事")
            return
        item = self._reopen_item(text)
        if item is not None and self._saved_referral is not None:
            # 保存の後に来訪者の側から触れた: その項目だけ聞き取りを再開する（LLM には聞かせない。
            # 「保存の後は氏名・電話を尋ねる文を捨てる」規則の、来訪者から触れたときだけの例外）
            ref_id, day, _receipt = self._saved_referral
            phase = {"phone": "ask_phone", "name": "ask_name", "area": "ask_area"}[item]
            self._pending = Pending("refer", text, language, turn_index=index, speaker=self._speaker,
                                    phase=phase, reopen=(ref_id, day), contact_pref="電話" if item == "phone" else "未定")
            logger.info("保存の後に来訪者が %s に触れたので、その項目だけ伺い直します", item)
            ask = {"phone": "ask_phone", "name": "ask_name", "area": "ask_area"}[item]
            line = self._t("refer", "reopen") + self._t("refer", ask)
            self.dialogue.add_turn(text, line)
            yield line
            return

        c = self._clarify_needed(text)
        if c is not None:
            async for s in self._ask_clarify(c, text, language, index):
                yield s
            return
        async for s in self._answer(text, language, index):
            yield s

    async def _answer(self, text: str, language: str, index: int, llm_text: str | None = None) -> AsyncIterator[str]:
        """ふつうの応答（LLM）と、それに付ける行為の提案。llm_text は LLM に渡す発話（聞き分けの短い返事の言い換え）。"""
        assert self._session is not None
        reply: list[str] = []
        offered = self._tidbit_to_offer(text)
        hint = self._tidbit_cfg.offer_template.format(text=offered["text"]) if offered else ""
        # 問いの文から後は、提案が付くか分かるまで（応答の終わりまで）読み上げずに預かる。
        # 提案が付けば LLM の問いは落とし、提案文だけを問いとして残す（問いは 1 回の応答で 1 つ）
        held: list[str] = []
        async for s in self.dialogue.respond(llm_text or text, language, hint=hint, speaker=self._speaker):
            reply.append(s)
            if held or self.dialogue.is_question(s):
                held.append(s)
                continue
            yield s
        if offered and self._session and self._tidbit_used(offered["text"], "".join(reply)):
            self._session.tidbit_used = True
            logger.info("さっきの方の小ネタを紹介しました")
        kind = (self.dialogue.last_proposal
                if self.dialogue.last_safety in ("complaint", "cannot_apply", "cannot_meet")  # 定型が付けた「伝えましょうか」
                else self._adjust_proposal(self.dialogue.last_proposal, text, "".join(reply)))
        procedure_turn = self._update_topic(text, kind)
        # 雑談のターン: 手続きの語が無く、案内役に何かを尋ねていない（「うんまだ見てないです」のような返事・報告も雑談に数える。
        # 天気などの語だけで数えていたので、紅葉の話への返事を数えられなかった）
        smalltalk = not procedure_turn and (
            self._smalltalk_re.search(text) is not None or re.search(self._cfg.share_fallback_topic, text) is not None
            or not self._visitor_question_re.search(text.strip()))
        self._session.smalltalk_streak = self._session.smalltalk_streak + 1 if smalltalk else 0
        s_ = self._session
        refer_offered = any(x.get("action") == "refer" for x in s_.proposals)
        if (kind is None and s_.topic_approved and s_.topic_no_refer >= self._news_cfg.topic_turns
                and not refer_offered and self._available("refer")):
            # 承認済みの手続きが話題なのに LLM が続けて紹介状を提案しなかった: 規則で提案する（お知らせより先）
            logger.info("承認済みの手続きの話が %d ターン続いたので、規則で紹介状を提案します", s_.topic_no_refer)
            kind = "refer"
        if (kind is None and self._news_cfg.propose_in_smalltalk
                and s_.smalltalk_streak >= self._news_cfg.smalltalk_turns
                and s_.turns >= self._news_cfg.smalltalk_min_session_turns):
            kind = "news"  # 雑談が続いた: 締めの定型の代わりに、お知らせを一つ紹介してよいか尋ねる
        proposing = kind is not None and self._available(kind)
        spoken = reply[: len(reply) - len(held)]  # 預からずに読み上げた文（応答の頭）
        for s in held:
            if proposing and self.dialogue.is_question(s):
                logger.info("提案文を付けるので、LLM の応答の問いの文を読み上げずに落としました")
                self.dialogue.note_drop("提案文の前の問い", s)
                continue
            spoken.append(s)
            yield s
        if len(spoken) < len(reply):
            # 読み上げなかった問いは履歴にも残さない
            self.dialogue.replace_last_reply((llm_text or text).strip(), "".join(reply), "".join(spoken))
        if kind == "news" and proposing:
            yield self._propose_news(text, language, index, "雑談の区切り", extend=True)
        elif kind is not None and proposing:
            line = self._t(kind, "propose")
            self.dialogue.extend_last_reply(line)
            self._pending = Pending(kind, text, language, turn_index=index, speaker=self._speaker,
                                    history_before=self.dialogue.history)
            self._session.proposals.append({"action": kind, "outcome": None})
            self.turn_info["proposed"] = kind
            logger.info("行為を提案しました: %s", kind)
            yield line

    # ------------------------------------------------------------------
    # 用件の聞き分け（WP13 E）
    # ------------------------------------------------------------------

    def _clarify_needed(self, text: str) -> Clarify | None:
        """聞き分けの問いが要るか（手続きの型で、候補が同程度に当たる。急いでいる・深刻な打ち明けの後・同じ組を聞いた後は聞かない）。"""
        s = self._session
        if s is None or s.news_blocked is not None or self.dialogue.intent_of(text) != "procedure":
            return None
        if self.dialogue.safety_reply(text) is not None:
            self.dialogue.last_safety = None  # 定型（代わりに申請・不満など）のターンでは聞かない（定型は応答の側で言う）
            return None
        c = self._clarifier.find(text)
        if c is None or frozenset(c.ids) in s.clarified:
            return None
        return c

    async def _ask_clarify(self, c: Clarify, text: str, language: str, index: int) -> AsyncIterator[str]:
        assert self._session is not None
        self._session.clarified.add(frozenset(c.ids))
        self._session.smalltalk_streak = 0
        line = self._clarifier.question(c)
        self._pending = Pending("clarify", text, language, phase="clarify", turn_index=index, speaker=self._speaker,
                                clarify=c)
        self.turn_info["clarify"] = f"聞き分けの問い（候補 {len(c.ids)} 件: {'・'.join(c.ids)}）"
        logger.info("用件の聞き分けを聞きます（候補 %d 件: %s）", len(c.ids), "・".join(c.ids))
        self.dialogue.add_turn(text, line)
        yield line

    def _clarify_done(self, c: Clarify, outcome: str) -> None:
        """聞き分けの結果を、文字記録（このターンの行為）と来訪記録に 1 行（本文は入れない）。"""
        note = f"聞き分けた（候補 {len(c.ids)} 件 → {outcome}）"
        self.turn_info["clarify"] = note
        logger.info("%s", note)
        if self._session is not None:
            self._session.proposals.append({"action": "clarify", "outcome": outcome, "candidates": str(len(c.ids))})

    async def _on_clarify_reply(self, text: str, language: str, index: int) -> AsyncIterator[str]:
        p = self._pending
        assert p is not None and p.clarify is not None
        c = p.clarify
        got = self._clarifier.pick(c, text)
        if isinstance(got, ProcedureRecord):
            # 決まった: その 1 件だけを資料にして、ふつうの流れで答える（承認済みならその中身、下書きなら担当の課で）
            self._pending = None
            self._clarify_done(c, got.id)
            self._clarified = got
            self.dialogue.forced_procedure = got
            # 短い返事（「二つ目」「やめるほう」）は、LLM には決まった手続きの名前で渡す（おうむ返しで「二つ目ですね」と言った）
            short = len(re.sub(r"[\s、。！？!?]", "", text)) <= self._cfg.reply_max_chars
            llm_text = f"{self._clarifier.label(got)}のことです。" if short else None
            async for s in self._answer(text, language, index, llm_text):
                yield s
            return
        if got == OTHER:
            self._pending = None
            self._clarify_done(c, "ほか")
            line = self._clarifier.rules.other
            self.dialogue.add_turn(text, line)
            yield line
            return
        if (classify_reply(text, self._words) == "unclear"
                and len(re.sub(r"[\s、。！？!?]", "", text)) > self._cfg.reply_max_chars
                and c.reasks >= self.dialogue._proc_cfg.clarify_max_reasks):
            # 問いに答えず別の話を始めた（聞き直しの後）: 聞き分けは閉じて、その話に答える
            self._pending = None
            self._clarify_done(c, "未決（別の話）")
            async for s in self._answer(text, language, index):
                yield s
            return
        if c.reasks < self.dialogue._proc_cfg.clarify_max_reasks:
            c.reasks += 1
            line = self._clarifier.reask(c)
            self.dialogue.add_turn(text, line)
            yield line
            return
        # 聞き直しても決まらない: 担当課は未特定のまま、紹介状を提案する
        self._clarify_done(c, "未決")
        line = self._clarifier.rules.fallback + self._t("refer", "propose")
        self._pending = Pending("refer", c.key, language, turn_index=index, speaker=self._speaker,
                                history_before=self.dialogue.history, unspecified=True)
        assert self._session is not None
        self._session.proposals.append({"action": "refer", "outcome": None})
        self.turn_info["proposed"] = "refer"
        self.dialogue.add_turn(text, line)
        yield line

    def _is_new_topic(self, text: str) -> bool:
        p = self._pending
        return (p is not None and p.phase == "proposed"
                and classify_reply(text, self._words) == "unclear"
                and len(re.sub(r"[\s、。！？!?]", "", text)) > self._cfg.reply_max_chars)

    def _adjust_proposal(self, kind: str | None, text: str, reply: str) -> str | None:
        """LLM のタグを規則で補正する（手続きの相談は lookup より refer、付け忘れの補い）。"""
        # 聞き分けで手続きが決まったターン（「入りたいんです」）も手続きの相談（WP13 E）
        procedure = re.search(self._ref_cfg.procedure_pattern, text) is not None or self.dialogue.forced_this_turn
        if kind == "lookup" and procedure and self._available("refer"):
            logger.info("手続き・相談の発話のため、提案を lookup から refer に置き換えます")
            return "refer"
        if kind == "refer" and not procedure and re.search(self._cfg.share_fallback_topic, text):
            # 季節・畑・海の雑談に付いた refer は合わない（覚えておくに合えばそちら）
            logger.info("地域の雑談に付いた refer の提案を捨てます")
            return "share" if self._share_fits(text) else None
        if (kind == "refer" and not procedure and re.search(self._cfg.lookup_fallback_topic, text)
                and self._available("lookup")):
            logger.info("行事・営業の質問なので、提案を refer から lookup に置き換えます")
            return "lookup"
        if kind == "share" and self._known_tidbit(text):
            logger.info("もう預かっている小ネタと同じ話なので share の提案を捨てます")
            return None
        if (kind == "lookup" and not re.search(self._cfg.question_pattern, text.strip())
                and not re.search(self._cfg.lookup_fallback_topic, text)):
            logger.info("尋ねていない発話への lookup の提案を捨てます")
            return None
        if (kind == "refer" and not procedure and not self._is_worry(text)
                and not re.search(self._cfg.refer_fallback_cue, text) and not re.search(self._cfg.worry_cue_pattern, text)):
            # 手続きでも困りごと・不安でもない発話（「あなたは職員さんですか」）に付いた refer は合わない
            logger.info("手続き・困りごとの無い発話に付いた refer の提案を捨てます")
            return None
        refer_offered = self._session is not None and any(x.get("action") == "refer" for x in self._session.proposals)
        if kind != "refer" and self._is_worry(text) and self._available("refer") and not refer_offered:
            # 健康・物忘れ・介護の不安は、雑談や調べるにせず専門の窓口につなぐ
            logger.info("健康・物忘れ・介護の不安の打ち明けなので refer を提案します")
            return "refer"
        if kind is None and self.dialogue.last_intent == "tourism" and self._available("lookup"):
            # 観光の問い（WP13 B）: 毎ターンの検索では資料を渡さないので、「調べてみましょうか？」を優先する
            logger.info("観光の問いなので lookup を提案します")
            return "lookup"
        if kind is not None:
            return kind
        if procedure and re.search(self._cfg.refer_fallback_cue, text) and self._available("refer"):
            logger.info("タグはありませんが、手続きの相談なので refer を提案します")
            return "refer"
        if self._share_fits(text):
            logger.info("タグはありませんが、地域の季節の話なので share を提案します")
            return "share"
        if (not procedure and re.search(self._cfg.lookup_fallback_reply, reply)
                and re.search(self._cfg.lookup_fallback_topic, text) and self._available("lookup")):
            logger.info("タグはありませんが、資料に無い行事・営業の質問なので lookup を提案します")
            return "lookup"
        return None

    def _is_worry(self, text: str) -> bool:
        """健康・物忘れ・介護の不安の打ち明けか（規則で。LLM に判断させない）。"""
        cfg = self._cfg
        return (re.search(cfg.worry_pattern, text) is not None
                or (re.search(cfg.worry_topic_pattern, text) is not None
                    and re.search(cfg.worry_cue_pattern, text) is not None))

    def _worry_department(self) -> str | None:
        """不安の打ち明けの担当課（地域包括支援センターを handles に持つ課、無ければ社会福祉課）。"""
        depts = self._facility.departments if self._facility else ()
        for d in depts:
            if any(self._cfg.worry_department_handle in h for h in d.handles):
                return d.name
        return next((d.name for d in depts if d.name == self._cfg.worry_fallback_department), None)

    def _share_fits(self, text: str) -> bool:
        """覚えておくの提案に合う雑談か（尋ねていない・手続きでない・個人の事情の語が無い・1 会話 1 回）。

        地域の季節の話題か、案内役が「教えていただけますか」と聞き返したのに来訪者が答えたとき（WP05）。
        """
        if not self._available("share") or self._session is None:
            return False
        if any(p.get("action") == "share" for p in self._session.proposals):
            return False
        taught = bool(self._turns) and re.search(self._cfg.teach_request_pattern, self._turns[-1][1]) is not None
        long_enough = len(re.sub(r"[\s、。！？!?]", "", text)) > self._cfg.reply_max_chars
        return ((re.search(self._cfg.share_fallback_topic, text) is not None or (taught and long_enough))
                and not self._known_tidbit(text)
                and not re.search(self._cfg.question_pattern, text.strip())
                and not re.search(self._ref_cfg.procedure_pattern, text)
                and not re.search(self._tidbit_cfg.banned_pattern, text))

    def _available(self, kind: str) -> bool:
        if kind == "news":
            return self._news_available()
        if kind not in self._cfg.enabled or kind not in self._templates:
            return False
        if kind == "lookup":
            return self._lookup is not None and bool(self._lookup.sources)
        if kind == "refer":
            return self._referrals is not None
        return kind == "share" and self._tidbits is not None

    def _t(self, action: str, key: str, **kw: Any) -> str:
        # 定型文の {greeting} は応答・相槌と同じ時刻判定で埋める
        return self._templates.get(action, {}).get(key, "").format(greeting=greeting_for(self._now()), org=self._org, **kw)

    def _outcome(self, outcome: str) -> None:
        self.turn_info["outcome"] = outcome
        if self._session and self._session.proposals:
            self._session.proposals[-1]["outcome"] = outcome

    async def _on_reply(self, text: str) -> AsyncIterator[str]:
        p = self._pending
        assert p is not None
        label = _PRIVATE_LABELS.get(p.phase)  # 答えを聞く前の段で決める（答えると段が進む）
        lines: list[str] = []
        if p.phase in ("proposed", "intake"):
            p.replies.append(text)
        if p.phase in _FREE_ANSWER_PHASES:
            async for s in self._on_intake_answer(p, text):
                lines.append(s)
                yield s
            user = self._private_text(p, text, label)
            if self._held and not lines:
                return  # 電話番号の断片は、そろってから 1 ターンとして積む
            self._add_turn(p, user, self.redact("".join(lines)))
            return
        verdict = classify_reply(text, self._words)
        if verdict == "unclear":
            if p.reasks < self._cfg.max_reask:
                p.reasks += 1
                line = self._reask_line(p)
                yield line
                self.dialogue.add_turn(text, line)
                return
            verdict = "no"  # 聞き直しても分からなければ辞退扱い
        if p.phase == "proposed":
            if verdict == "no":
                self._outcome("declined")
                self._pending = None
                line = self._t(p.action, "declined")
                yield line
                self.dialogue.add_turn(text, line)
                return
            self._outcome("accepted")
            p.consent_speaker = self._speaker
            async for s in self._act(p):
                lines.append(s)
                yield s
        elif p.phase == "confirm_name":
            async for s in self._on_confirm_name(p, verdict == "yes"):
                lines.append(s)
                yield s
        elif p.phase == "confirm_phone":
            if verdict == "no" and p.contact_pref == "電話" and not p.phone_reasked:
                # 連絡方法が電話: 番号がそろうまで保存しない（1 回だけ言い直してもらう）
                p.phone, p.phone_reasked, p.phase = None, True, "ask_phone"
                line = self._t("refer", "phone_reask")
                lines.append(line)
                yield line
            else:
                prefix = ""
                if verdict == "no":
                    p.phone = None
                    prefix = self._t("refer", "phone_at_counter" if p.contact_pref == "電話" else "phone_wrong")
                async for s in self._finish_referral(p, prefix=prefix):
                    lines.append(s)
                    yield s
        elif p.phase == "confirm_contact":
            if verdict == "yes":
                p.contact_pref = "電話"
            else:
                # 「いいえ、待ちます」のように別の方法を言っていればそれ、無ければ未定
                others = [c for c in self._contact_choices(text) if c != "電話"]
                p.contact_pref = others[0] if len(others) == 1 else "未定"
            async for s in self._after_contact(p, text if verdict == "yes" else ""):
                lines.append(s)
                yield s
        elif p.phase == "confirm_share":
            async for s in self._finish_share(p, verdict == "yes"):
                lines.append(s)
                yield s
        user = self._private_text(p, text, label)
        if p.action == "news" and self._news_history is not None:
            # お知らせを紹介した: 履歴には要約の本文ではなく定型の一言（WP07 と同じ）
            lines, self._news_history = [self._news_history], None
        self._add_turn(p, user, self.redact("".join(lines)))

    def _add_turn(self, p: Pending, user: str, assistant: str) -> None:
        """返事待ちのターンを履歴に積む。紹介状を保存した直後なら、聞き取りを外して 1 行にまとめる。"""
        if self._history_note is not None:
            self.dialogue.restore(p.history_before, *self._history_note)
            self._history_note = None
            logger.info("紹介状の聞き取りを対話履歴から外し、受付の 1 行だけを残しました")
            return
        self.dialogue.add_turn(user, assistant)

    def _reask_line(self, p: Pending) -> str:
        if p.phase == "confirm_contact":
            return self._t("refer", "confirm_contact")
        if p.phase == "confirm_phone":
            return self._t("refer", "confirm_phone", phone=phone_for_speech(p.phone or ""))
        if p.phase == "confirm_name":
            return self._t("refer", "confirm_name", name=p.name or "")
        if p.phase == "confirm_share":
            return self._t("share", "confirm_text", text=p.share_text or "")
        return self._t(p.action, "reask")

    async def _act(self, p: Pending) -> AsyncIterator[str]:
        if p.action in ("lookup", "refer"):
            yield self._t(p.action, "acting")  # refer は「担当の方が準備できるように、少しだけ伺いますね」
            if self._on_state is not None:
                await self._on_state("thinking")
        elif self._on_state is not None and p.action != "news":
            await self._on_state("thinking")  # 書き直しの間（言葉は挟まない。すぐ確認の文が出る）
        if p.action == "news":
            async for s in self._do_news(p):
                yield s
        elif p.action == "lookup":
            self._pending = None
            async for s in self._do_lookup(p):
                yield s
        elif p.action == "refer":
            async for s in self._do_refer(p):
                yield s
        elif p.action == "share":
            async for s in self._do_share(p):
                yield s

    # ---- お知らせ（WP11） --------------------------------------------------

    def _news_block_check(self, text: str) -> None:
        """この会話でお知らせを提案しない理由（相手が急いでいる語）を覚える。深刻な打ち明けは定型の側で。"""
        if self._session is not None and self._session.news_blocked is None and self._hurry_re.search(text):
            self._session.news_blocked = "急いでいる"
            logger.info("相手が急いでいるので、この会話ではお知らせを提案しません")

    def _news_available(self) -> bool:
        """お知らせを提案してよいか（返事待ち・聞き取りの途中でない、止める理由が無い、1 会話の上限、候補がある）。"""
        s = self._session
        if (self._news is None or not self._news_cfg.enabled or s is None or self._pending is not None
                or s.news_blocked is not None or s.news_proposed >= self._news_cfg.max_per_session):
            return False
        return self._news.next_candidate() is not None

    def _is_news_ask(self, text: str) -> bool:
        """来訪者が直接お知らせ・催しを聞いたか（問い・頼みの形で、手続きの語が無い）。深刻な打ち明けの後の会話では見ない。"""
        t = text.strip()
        if self._news is None or not self._news_cfg.enabled:
            return False
        if self._session is not None and self._session.news_blocked == "深刻な打ち明け・迷子":
            return False
        return (self._news_ask_re.search(t) is not None and self._news_ask_request_re.search(t) is not None
                and not re.search(self._ref_cfg.procedure_pattern, t))

    async def _present_news_direct(self, text: str, language: str, index: int) -> AsyncIterator[str]:
        """聞かれたのでお知らせを 1 件紹介する（提案文なし）。候補が無い・この会話で紹介済みなら定型の一言。"""
        s = self._session
        assert s is not None
        shown = any(x.get("action") == "news" and x.get("outcome") == "shown" for x in s.proposals)
        if shown or self._news is None or self._news.next_candidate() is None:
            line = self._t("news", "none_direct")
            logger.info("お知らせを聞かれましたが、%s", "この会話では紹介済みです" if shown else "候補がありません")
            self.dialogue.add_turn(text, line)
            yield line
            return
        s.news_proposed += 1
        s.proposals.append({"action": "news", "outcome": None, "trigger": "直接聞かれた"})
        self.turn_info["proposed"] = "news"
        logger.info("お知らせを直接聞かれたので、すぐ紹介します")
        p = Pending("news", text, language, turn_index=index, speaker=self._speaker,
                    history_before=self.dialogue.history)
        self._pending = p
        head = self._t("news", "direct")
        yield head
        async for line in self._do_news(p):
            yield line
        self.dialogue.add_turn(text, head + (self._news_history or ""))
        self._news_history = None

    def _update_topic(self, text: str, kind: str | None) -> bool:
        """話題の手続きを覚える（このターンで手続きを言ったら True）。手続きでない問いが来たら話題は終わり。"""
        s = self._session
        assert s is not None
        book = self._procedures.book
        matched = self.dialogue.last_procedures
        mentioned = sorted(self._procedures.mentioned(text, self._facility))
        key = matched[0].id if matched else (mentioned[0] if mentioned else None)
        if key is not None:
            if key != s.topic:
                s.topic, s.topic_turns, s.topic_no_refer = key, 0, 0
                s.topic_approved = any(book is not None and book.usable(r) for r in matched)
            s.topic_turns += 1
        elif s.topic is not None and self._visitor_question_re.search(text.strip()):
            s.topic, s.topic_turns, s.topic_approved, s.topic_no_refer = None, 0, False, 0
        elif s.topic is not None:
            s.topic_turns += 1  # 「退職とかで」のような、手続きの話の続き
        if s.topic is not None and s.topic_approved:
            s.topic_no_refer = s.topic_no_refer + 1 if kind != "refer" else 0
        return key is not None

    def _news_status(self) -> str:
        """このターンのお知らせの判定（ログ用。提案したか、出さなかった理由、雑談のターン数、話題の手続き）。"""
        s = self._session
        if s is None:
            return "会話なし"
        if self.turn_info.get("proposed") == "news":
            state = "提案した（または紹介した）"
        elif self._news is None or not self._news_cfg.enabled:
            state = "出さない（お知らせなし・無効）"
        elif s.news_blocked is not None:
            state = f"出さない（{s.news_blocked}）"
        elif s.news_proposed >= self._news_cfg.max_per_session:
            state = "出さない（この会話で提案済み）"
        elif self._pending is not None:
            state = f"出さない（返事待ち・聞き取り中: {self._pending.action}/{self._pending.phase}）"
        elif self._news.next_candidate() is None:
            state = "出さない（候補なし）"
        else:
            state = "きっかけ待ち"
        topic = f"{s.topic}({s.topic_turns}ターン{'・承認済み' if s.topic_approved else ''})" if s.topic else "なし"
        return f"{state} / 雑談 {s.smalltalk_streak} ターン / 会話 {s.turns} ターン / 話題の手続き {topic}"

    def _propose_news(self, text: str, language: str, index: int, trigger: str, extend: bool = True) -> str:
        """お知らせの提案（決まった文）。返事待ちにして、履歴の直前の応答の後ろに足す。"""
        assert self._session is not None
        line = self._t("news", "propose")
        if extend:
            self.dialogue.extend_last_reply(line)
        self._pending = Pending("news", text, language, turn_index=index, speaker=self._speaker,
                                history_before=self.dialogue.history)
        self._session.news_proposed += 1
        self._session.proposals.append({"action": "news", "outcome": None, "trigger": trigger})
        self.turn_info["proposed"] = "news"
        logger.info("お知らせの紹介を提案しました（%s）", trigger)
        return line

    @property
    def pending_action(self) -> str | None:
        """返事を待っている行為（無ければ None）。"""
        return self._pending.action if self._pending is not None else None

    async def news_timeout(self) -> AsyncIterator[str]:
        """お知らせの提案に無言のまま reply_timeout_sec たった: 「わかりました。」と言って、そのまま（再提案しない）。"""
        p = self._pending
        if p is None or p.action != "news" or p.phase != "proposed":
            return
        self._pending = None
        self._outcome("declined")
        line = self._t("news", "declined")
        self.dialogue.extend_last_reply(line)
        logger.info("お知らせの提案に返事が無いので、紹介しません")
        yield line

    def take_news_closing(self) -> str | None:
        """紹介の要約を読み終えた後に言う締めの文（main が 2 秒置いて言い、構図を戻す）。1 回だけ返す。"""
        line, self._news_closing = self._news_closing, None
        return line

    async def _do_news(self, p: Pending) -> AsyncIterator[str]:
        self._pending = None
        rec = self._news.next_candidate() if self._news is not None else None
        if rec is None:
            self._outcome("none")
            yield self._t("news", "none")
            return
        assert self._news is not None and self._session is not None
        self._news.mark_shown(rec.id)
        self._session.proposals[-1]["news_id"] = rec.id  # 来訪記録には記事の id だけ（本文は入れない）
        self._outcome("shown")
        logger.info("お知らせを紹介します: %s（%s）", rec.id, rec.category)
        if self._on_layout is not None:
            await self._on_layout(self._news.layout_message(rec))
        intro = self._t("news", "intro", when=spoken_date(rec.published_date(), self._now().date(), self._news_cfg))
        closing = self._t("news", "closing")
        self._news_closing = closing
        self._news_history = self._t("news", "history", intro=intro, title=rec.title, closing=closing)
        yield intro
        for sentence in re.split(r"(?<=[。！？!?])", tidy_summary(rec.summary)):
            if sentence.strip():
                yield sentence.strip()

    # ---- 調べる --------------------------------------------------------

    async def _do_lookup(self, p: Pending) -> AsyncIterator[str]:
        assert self._lookup is not None
        try:
            result = await self._lookup.run(p.user_text)
        except Exception:
            logger.exception("調べる途中で失敗しました")
            yield self._t("lookup", "failed")
            return
        if not result.found:
            yield self._t("lookup", "not_found" if result.queued else "failed")
            return
        async for s in self.dialogue.respond(p.user_text, p.language, extra_refs=result.references,
                                             speaker=p.speaker):
            yield s

    # ---- つなぐ（紹介状）------------------------------------------------

    async def _do_refer(self, p: Pending) -> AsyncIterator[str]:
        """同意の後: 担当課を規則で決め、手続きの項目（最大 3 問）と共通の項目を順に伺う。"""
        window = self._refer_window(p)
        # 担当課は規則だけで決める（提案のきっかけと同意の返事。決まらなければ話題全体）。
        # 一意に決まらなければ「未特定」で保存し、職員が職員パネルで課を付ける。来訪者には聞き返さない
        recent = "\n".join([p.user_text, *p.replies])
        topic = "\n".join(u for u, _ in window)
        cands = (self._procedures.departments_for(recent, self._facility)
                 or self._procedures.departments_for(topic, self._facility))
        worry = self._worry_department() if self._is_worry(topic) else None
        decided = self._clarified
        book = self._procedures.book
        decided_ok = decided is not None and book is not None and book.usable(decided) and bool(decided.department)
        if p.unspecified:
            # 聞き分けで決まらなかった: 規則でも決めず、職員が職員パネルで課を付ける（WP13 E-3）
            cands = []
            worry = None
        elif decided_ok and decided is not None:
            cands = [decided.department]  # 聞き分けで決まった承認済みの手続きの課
        if len(cands) == 1:
            p.department = cands[0]
        elif worry is not None:
            # 健康・物忘れ・介護の不安: 地域包括支援センターを持つ課（無ければ社会福祉課）
            p.department = worry
            logger.info("健康・物忘れ・介護の不安の相談なので %s につなぎます", worry)
        else:
            p.department = UNSPECIFIED_DEPARTMENT
            logger.info("担当課が規則で一意に決まりません（候補 %d）。未特定で保存します", len(cands))
        p.phone = find_phone(topic)  # 来訪者が先に言っていれば、後で読み上げて確かめる
        if p.unspecified:
            p.intake_set = None
        elif decided_ok and decided is not None:
            p.intake_set = decided.intake_set()
        else:
            p.intake_set = (self._procedures.intake_for(recent, self._facility)
                            or self._procedures.intake_for(topic, self._facility))
        if p.intake_set is not None:
            # 聞き分けの 1 問と聞き取りは合わせて最大 3 問（聞き分けで決まったら 1 問減らす。WP13 E-1）
            most = self._ref_cfg.intake_max_questions - (1 if decided is not None else 0)
            labels = list(p.intake_set.questions[: max(0, most)])
            asks = await self._phrase_questions(labels, p.user_text)
            p.intake = [{"question": q, "ask": a, "answer": ""} for q, a in zip(labels, asks, strict=True)]
        if p.intake:
            p.phase, p.intake_idx = "intake", 0
            yield p.intake[0]["ask"]
            return
        p.phase = "ask_name"
        yield self._t("refer", "ask_name")

    async def _phrase_questions(self, labels: list[str], user_text: str) -> list[str]:
        """聞き取りの項目を、LLM に自然な言い方の質問にしてもらう（だめなら決まった形）。"""
        fallback = [f"{_PAREN_RE.sub(lambda m: '（' + m.group(1) + 'など）', q)}を教えていただけますか？"
                    for q in labels]
        prompt = self._ref_cfg.intake_phrase_prompt
        body = f"来訪者の発話: {user_text}\n項目:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(labels))
        data = await self._chat_json(prompt, body, self._ref_cfg.intake_phrase_max_tokens)
        got = (data or {}).get("questions")
        if not isinstance(got, list) or len(got) != len(labels):
            return fallback
        out = []
        for q, fb in zip(got, fallback, strict=True):
            q = str(q).strip()
            ok = 4 <= len(q) <= 60 and q[-1] in "？?か" and not re.search(r"[A-Za-z]{3,}", q)
            out.append(q if ok else fb)
        return out

    async def _on_intake_answer(self, p: Pending, text: str) -> AsyncIterator[str]:
        """自由な答えを待つ段（手続きの項目・お名前・地区・連絡方法・電話番号）。答えなくても先へ進む。"""
        cfg = self._ref_cfg
        skipped = re.search(cfg.skip_pattern, text) is not None
        ack = self._t("refer", "skipped") if skipped else ""
        if p.phase == "intake":
            answer = "" if skipped else text.strip()
            if answer and self._speaker and p.consent_speaker and self._speaker != p.consent_speaker:
                answer += cfg.companion_answer_suffix  # 付き添いの方が代わりに答えた（職員に分かるように）
            p.intake[p.intake_idx]["answer"] = answer
            p.intake_idx += 1
            if p.intake_idx < len(p.intake):
                yield ack + p.intake[p.intake_idx]["ask"]
                return
            p.phase = "ask_name"
            yield ack + self._t("refer", "ask_name")
        elif p.phase == "ask_name":
            name = None if skipped else _clean_name(text, cfg.name_strip_pattern, cfg.name_max_chars)
            if name:
                p.name, p.phase = name, "confirm_name"
                yield self._t("refer", "confirm_name", name=name)
                return
            p.name = None
            async for s in self._ask_area(p, ack):
                yield s
        elif p.phase == "ask_area":
            if not skipped:
                area = re.sub(r"(です|に住んで(い)?ます|に住んでる|から来ました|あたり|のほう)[。、]?$", "",
                              text.strip()).strip("。、 ")
                if not self._full_address(p):
                    area = re.sub(cfg.address_number_pattern, "", area).strip("。、 ")
                p.address_area = area[:40] or None
            if p.reopen is not None:
                async for s in self._finish_referral(p, prefix=ack):  # 保存の後にお住まいだけ伺い直した
                    yield s
                return
            p.phase = "ask_contact"
            yield ack + self._t("refer", "ask_contact")
        elif p.phase == "ask_contact":
            matched = self._contact_choices(text)
            phone = find_phone(text)
            if phone:
                p.phone = phone
            if phone or phone_digits(text):
                matched = ["電話"]  # 番号を言い始めた（「090の」）なら電話
            if len(matched) != 1:
                # 判定できない（どれにも当たらない・2 つ以上に当たる）: 電話か確かめる
                logger.info("連絡方法を判定できないので、お電話かを確かめます（候補 %d）", len(matched))
                p.phase, p.reasks = "confirm_contact", 0
                yield self._t("refer", "confirm_contact")
                return
            p.contact_pref = matched[0]
            async for s in self._after_contact(p, text):
                yield s
        elif p.phase == "ask_phone":
            async for s in self._collect_phone(p, text, skipped):
                yield s

    def _contact_choices(self, text: str) -> list[str]:
        return list(dict.fromkeys(v for pat, v in self._ref_cfg.contact_rules if re.search(pat, text)))

    async def _after_contact(self, p: Pending, text: str = "") -> AsyncIterator[str]:
        """連絡方法が決まった後: 電話なら番号がそろうまで保存しない。それ以外は保存する。"""
        if p.phone:
            p.phase, p.reasks = "confirm_phone", 0
            yield self._t("refer", "confirm_phone", phone=phone_for_speech(p.phone))
        elif p.contact_pref == "電話" and phone_digits(text):
            p.phase = "ask_phone"  # 番号を区切って言い始めた: 続きを待って連結する
            async for s in self._collect_phone(p, text, skipped=False):
                yield s
        elif p.contact_pref == "電話":
            p.phase = "ask_phone"
            yield self._t("refer", "ask_phone")
        else:
            async for s in self._finish_referral(p):
                yield s

    async def _collect_phone(self, p: Pending, text: str, skipped: bool) -> AsyncIterator[str]:
        """電話番号の断片を連結する。10〜11 桁にそろったら復唱、まだなら続きを待つ（何も言わない）。"""
        if skipped:
            # 言わない・分からない: 番号なしで保存して、窓口で伺うと伝える
            p.phone_buf, p.phone_deadline, p.phone = "", None, None
            logger.info("電話番号を言わないとのことなので、番号なしで紹介状を作ります")
            async for s in self._finish_referral(p, prefix=self._t("refer", "phone_at_counter")):
                yield s
            return
        p.phone_buf += phone_digits(text)
        phone = format_phone(p.phone_buf)
        if phone:
            p.phone, p.phase, p.reasks = phone, "confirm_phone", 0
            p.phone_buf, p.phone_deadline = "", None
            logger.info("電話番号の数字がそろったので復唱します")
            yield self._t("refer", "confirm_phone", phone=phone_for_speech(phone))
            return
        if len(p.phone_buf) > PHONE_DIGITS_MAX:
            logger.info("電話番号の数字が多すぎます（%d 桁）", len(p.phone_buf))
            async for s in self._phone_incomplete(p):
                yield s
            return
        p.phone_deadline = self._clock() + self._ref_cfg.phone_wait_sec
        self._held = True
        logger.info("電話番号の続きを待ちます（いま %d 桁、%.0f 秒）", len(p.phone_buf), self._ref_cfg.phone_wait_sec)

    async def _phone_incomplete(self, p: Pending) -> AsyncIterator[str]:
        """番号がそろわなかった: 1 回だけ聞き直し、それでもだめなら番号なしで紹介状を作る。"""
        p.phone_buf, p.phone_deadline = "", None
        if not p.phone_reasked:
            p.phone_reasked = True
            yield self._t("refer", "phone_reask")
            return
        logger.info("電話番号がそろわないので、番号なしで紹介状を作ります")
        p.phone = None
        async for s in self._finish_referral(p, prefix=self._t("refer", "phone_at_counter")):
            yield s

    @property
    def phone_deadline(self) -> float | None:
        """電話番号の続きを待っていれば、その期限（monotonic 秒）。"""
        p = self._pending
        return p.phone_deadline if p is not None and p.phase == "ask_phone" else None

    async def on_phone_timeout(self) -> AsyncIterator[str]:
        """続きを待つ期限を過ぎたら、聞き直すか番号なしで進める（期限前なら何もしない）。"""
        p = self._pending
        deadline = self.phone_deadline
        if p is None or deadline is None or self._clock() < deadline:
            return
        self.dialogue.begin_turn()
        logger.info("電話番号の続きが %.0f 秒来ませんでした（%d 桁）", self._ref_cfg.phone_wait_sec, len(p.phone_buf))
        lines: list[str] = []
        async for s in self._phone_incomplete(p):
            lines.append(s)
            yield s
        self._add_turn(p, f"（{_PRIVATE_LABELS['ask_phone']}を伺いました）", self.redact("".join(lines)))

    def ignores(self, text: str, heard_at: float | None = None) -> bool:
        """紹介状の保存中・保存直後に来た、数字だけ・1〜2 語だけの発話か（言い終わりの断片とみなして無視する）。

        heard_at はその発話を聞き終えた時刻（monotonic）。保存の途中に聞き取った断片は、保存の読み上げ（10 秒前後）が
        終わるまで順番を待つので、処理する時刻ではなく聞いた時刻で 10 秒を数える。
        """
        cfg = self._ref_cfg
        if self._pending is not None:
            return False
        if not self._saving:
            if self._saved_at is None:
                return False
            t = heard_at if heard_at is not None else self._clock()
            if t - self._saved_at > cfg.post_save_ignore_sec:
                return False
        digits_only = re.fullmatch(r"[0-9\s、。，．,.！？!?の\-ー－‐]+", normalize_digits(text)) is not None
        words = [w for w in re.split(r"[\s、。，．,.！？!?…]+", text) if w]
        short = 0 < len(words) <= cfg.post_save_fragment_max_words and len("".join(words)) <= cfg.post_save_fragment_max_chars
        return (digits_only and bool(phone_digits(text))) or short

    async def _on_confirm_name(self, p: Pending, ok: bool) -> AsyncIterator[str]:
        if ok and p.reopen is not None:
            async for s in self._finish_referral(p):  # 保存の後にお名前だけ伺い直した
                yield s
        elif ok:
            async for s in self._ask_area(p, ""):
                yield s
        elif not p.name_retry:
            p.name, p.name_retry, p.phase = None, True, "ask_name"
            yield self._t("refer", "ask_name_again")
        else:
            p.name = None  # 二度違えば控えない
            async for s in self._ask_area(p, self._t("refer", "name_dropped")):
                yield s

    async def _ask_area(self, p: Pending, prefix: str) -> AsyncIterator[str]:
        p.phase = "ask_area"
        yield prefix + self._t("refer", "ask_area_full" if self._full_address(p) else "ask_area")

    def _full_address(self, p: Pending) -> bool:
        return re.search(self._ref_cfg.full_address_pattern, "\n".join([p.user_text, *p.replies])) is not None

    async def _finish_referral(self, p: Pending, prefix: str = "") -> AsyncIterator[str]:
        """聞き取りを終えて紹介状を作り、受付番号と持ち物の案内を言う。"""
        assert self._referrals is not None
        self._pending = None
        self._saving = True  # 保存の途中に聞き取った番号の残りなども無視する（ignores）
        try:
            async for s in self._save_referral(p, prefix):
                yield s
        finally:
            self._saving = False

    async def _save_referral(self, p: Pending, prefix: str) -> AsyncIterator[str]:
        assert self._referrals is not None
        if p.reopen is not None:
            async for s in self._update_referral(p, prefix):
                yield s
            return
        if self._on_state is not None:
            await self._on_state("thinking")
        fields = await self._structure_referral(self._refer_window(p), p)
        try:
            ref = await asyncio.to_thread(self._referrals.create, fields, self._edge_id(), p.language)
        except Exception:
            logger.exception("紹介状を保存できません")
            yield prefix + self._t("refer", "failed")
            return
        self.last_receipt = ref.receipt
        self._saved_referral = (ref.id, ref.created_at[:10], ref.receipt)
        # 会話を仕切り直す: 聞き取りの質問と答えは履歴から外し、受付の 1 行だけを残す（_add_turn）
        self._saved_at = self._clock()
        unknown_dept = ref.department in ("", UNSPECIFIED_DEPARTMENT)
        self._history_note = (self._ref_cfg.history_note_user,
                              self._t("refer", "history_note", number=ref.receipt,
                                      department="担当の方" if unknown_dept else ref.department))
        if self._session is not None:
            self._session.referral_id = ref.id
        logger.info("紹介状を保存しました: 受付 %s・%s・聞き取り %d 問", ref.receipt, ref.department,
                    sum(1 for q in p.intake if q["answer"]))
        unknown = ref.department in ("", UNSPECIFIED_DEPARTMENT)
        key = ("done_unknown" if unknown else "done") + ("" if ref.name else "_no_name")
        line = prefix + self._t("refer", key, department=ref.department, number=ref.receipt)
        answers = "\n".join([p.user_text, *(q["answer"] for q in p.intake)])
        item = p.intake_set.bring_for(answers) if p.intake_set else None
        if item:
            line += self._t("refer", "bring", item=item)
        self._business_done = True
        yield line

    async def _update_referral(self, p: Pending, prefix: str) -> AsyncIterator[str]:
        """保存の後に伺い直した項目だけを、同じ紹介状に書き足す。"""
        assert self._referrals is not None and p.reopen is not None
        ref_id, day = p.reopen
        fields: dict[str, str | None] = {"name": p.name, "phone": p.phone, "address_area": p.address_area}
        if p.phone:
            fields["contact_pref"] = "電話"
        if not any(fields.values()):
            yield prefix + self._t("refer", "update_skipped")
            return
        try:
            ref = await asyncio.to_thread(self._referrals.update_contact, ref_id, date.fromisoformat(day), **fields)
        except Exception:
            logger.exception("紹介状を書き直せません")
            ref = None
        if ref is None:
            yield prefix + self._t("refer", "failed")
            return
        logger.info("保存の後に伺った項目を紹介状に書き足しました: 受付 %s（%s）", ref.receipt,
                    "・".join(k for k, v in fields.items() if v))
        self._saved_at = self._clock()
        yield prefix + self._t("refer", "updated", number=ref.receipt)

    def _reopen_item(self, text: str) -> str | None:
        """保存の後に、来訪者がお名前・お電話番号・お住まいに触れたらその項目（無ければ None）。"""
        if self._saved_referral is None or self._referrals is None:
            return None
        return next((item for item, pat in self._ref_cfg.reopen_patterns if re.search(pat, text)), None)

    def _transcript(self, history: list[tuple[str, str]]) -> str:
        return "\n".join(f"来訪者: {u}\n案内役: {a}" for u, a in history)

    def _refer_window(self, p: Pending) -> list[tuple[str, str]]:
        """紹介状を作る範囲: 提案（refer）のきっかけの発話から後と、同意・聞き取りの答え。

        きっかけより前の往復は使わない（前の話題が要約に混ざったため。WP06）。
        """
        return self._turns[p.turn_index:p.turn_index + 1] + [(r, "") for r in p.replies]

    async def _structure_referral(self, history: list[tuple[str, str]], p: Pending) -> dict[str, Any]:
        """紹介状の項目を作る。要約・区分などは LLM、課・氏名・地区・連絡方法・電話番号は規則と聞き取りから。

        LLM に失敗しても、来訪者の言葉をそのまま要約にして作る（答えなくても紹介状は作る）。
        """
        prompt = (self._ref_cfg.structure_prompt
                  + "\nneeds_action は職員の対応が要るか（案内だけで済むなら false）。"
                  + "tags は【タグの一覧】から最大 5 つ選ぶ（一覧に無いものは作らない）。topic_free は話題を 1 行で。"
                  + '\n出力に "needs_action": true, "tags": [], "topic_free": "" も含める。'
                  + "\n【タグの一覧】" + "、".join(self._tags))
        # 来訪者の言葉と案内役の言ったことを分けて渡す（同意の「お願いします」だけの返事は来訪者の話に入れない）
        # 案内役のメモ・要約は、聞き取り間違いの対応表を当てた文から作る（「国宝」のまま職員に渡さない）
        visitor = [self._fix(u) for u, _ in history if u and not self._is_bare_reply(u)]
        assistant = [self._fix(a) for _, a in history if a]
        body = ("【来訪者の言葉】\n" + "\n".join(visitor or [p.user_text])
                + "\n【案内役が言ったこと】\n" + ("\n".join(assistant) or "なし"))
        data = await self._chat_json(prompt, body, self._ref_cfg.summary_max_tokens) or {}
        summary = self._fix(self._visitor_only(str(data.get("summary") or ""), "\n".join(visitor))
                            or p.user_text)
        note = self._fix(str(data.get("assistant_note") or "").strip().replace("\n", " "))
        # タグは、手続きが規則で分かれば procedures.yaml の手続き→タグだけ。分からないときだけ LLM の選んだタグ
        proc_tags = pick_tags(self._procedures.tags_for("\n".join(visitor) or p.user_text, self._facility), self._tags)
        tags = proc_tags or pick_tags(data.get("tags"), self._tags)
        return {
            "department": p.department or UNSPECIFIED_DEPARTMENT,
            "category": _pick(data.get("category"), CATEGORIES, "その他"),
            "summary": summary,
            "urgency": _pick(data.get("urgency"), URGENCIES, "通常"),
            "contact_pref": p.contact_pref if p.contact_pref in CONTACT_PREFS else "未定",
            "name": p.name,
            "phone": p.phone,
            "needs_action": bool(data.get("needs_action", True)),
            "tags": tags,
            "topic_free": str(data.get("topic_free") or "")[:60],
            "intake": [{"question": q["question"], "answer": q["answer"] or "（答えなし）"} for q in p.intake],
            "address_area": p.address_area,
            "assistant_note": note[: self._ref_cfg.assistant_note_max_chars],
        }

    def _fix(self, text: str) -> str:
        return self._asr_fixes.apply(text) if self._asr_fixes is not None and text else text

    def _is_bare_reply(self, text: str) -> bool:
        """「お願いします」「はい」だけの返事か（来訪者の話としては要約に入れない）。"""
        return (classify_reply(text, self._words) != "unclear"
                and len(re.sub(r"[\s、。！？!?]", "", text)) <= self._cfg.reply_max_chars)

    def _visitor_only(self, summary: str, visitor_text: str) -> str:
        """要約から、来訪者が言っていない課名を含む文と、案内役の側の経緯の文を落とす。"""
        depts = [d.name for d in (self._facility.departments if self._facility else ())]
        kept = []
        for sent in re.split(r"(?<=[。！？!?])", summary.strip()):
            if not sent.strip():
                continue
            if re.search(self._ref_cfg.summary_meta_pattern, sent):
                continue
            if any(d in sent and d not in visitor_text for d in depts):
                continue
            kept.append(sent.strip())
        return "".join(kept)

    # ---- 個人の情報を画面・ログに出さない ---------------------------------

    @property
    def private_label(self) -> str | None:
        """いまの返事が個人の情報か（字幕・コンソールでは「お名前：伺いました」のように伏せる）。"""
        p = self._pending
        return _PRIVATE_LABELS.get(p.phase) if p is not None else None

    @property
    def acting(self) -> bool:
        """紹介状の聞き取りの途中か（同意の後。提案の返事待ちは含まない）。"""
        p = self._pending
        return p is not None and p.action == "refer" and p.phase != "proposed"

    @property
    def expects_short_reply(self) -> bool:
        """はい・いいえの返事を待っているか（ASR の言語を固定する）。自由な答えの段は固定しない。"""
        return self._pending is not None and self._pending.phase not in _FREE_ANSWER_PHASES

    def redact(self, text: str) -> str:
        """画面・コンソールに出す文から、聞き取った氏名・地区・電話番号を伏せる。"""
        values = set(self._private_values)
        p = self._pending
        if p is not None:
            values |= {v for v in (p.name, p.address_area, p.phone,
                                   phone_for_speech(p.phone) if p.phone else None) if v}
        for v in sorted(values, key=len, reverse=True):
            text = text.replace(v, "＊＊")
        return text

    def _private_text(self, p: Pending, text: str, label: str | None) -> str:
        for v in (p.name, p.address_area, p.phone, phone_for_speech(p.phone) if p.phone else None):
            if v:
                self._private_values.add(v)
        return f"（{label}を伺いました）" if label else text

    async def _tag_conversation(self, history: list[tuple[str, str]]) -> list[str]:
        if not history or not self._tags:
            return []
        prompt = ("次の会話で来訪者が実際に話した話題を、【タグの一覧】から 1〜3 個（最大 5 個）選んで"
                  "JSON で出力してください。話に出ていないタグは選ばない。一覧に無いものは作らない。"
                  '出力: {"tags": []}\n【タグの一覧】' + "、".join(self._tags))
        data = await self._chat_json(prompt, self._transcript(history), 64)
        return pick_tags((data or {}).get("tags"), self._tags)

    async def _chat_json(self, system: str, user: str, max_tokens: int) -> dict[str, Any] | None:
        try:
            raw = await asyncio.to_thread(
                self._llm.chat, self._model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            logger.warning("LLM に構造化を頼めませんでした", exc_info=True)
            return None
        m = _JSON_RE.search(raw or "")
        try:
            data = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    # ---- 覚えておく（今日の小ネタ。WP04 C）-------------------------------

    async def _do_share(self, p: Pending) -> AsyncIterator[str]:
        """来訪者の話を匿名の一言に書き直し、読み上げて確かめる（保存は確認の後）。"""
        text = await self._rewrite_tidbit(p)
        if text is None:
            self._pending = None
            self._outcome("failed")
            yield self._t("share", "failed")
            return
        p.phase, p.share_text, p.reasks = "confirm_share", text, 0
        yield self._t("share", "confirm_text", text=text)

    async def _finish_share(self, p: Pending, ok: bool) -> AsyncIterator[str]:
        self._pending = None
        if not ok or self._tidbits is None or not p.share_text:
            self._outcome("declined")
            yield self._t("share", "declined")
            return
        try:
            item = self._tidbits.add(p.share_text)
        except OSError:
            logger.exception("小ネタを保存できませんでした")
            yield self._t("share", "failed")
            return
        if self._session is not None:
            self._session.own_tidbits.append(item["id"])
        logger.info("小ネタを預かりました: %s", item["id"])
        yield self._t("share", "done")

    async def _rewrite_tidbit(self, p: Pending) -> str | None:
        cfg = self._tidbit_cfg
        # 提案のきっかけになった発話と、その前の来訪者の発話（直近 2 つ）から書き直す
        users = [u for u, _ in self.dialogue.history][-2:]
        source = "\n".join(dict.fromkeys([*users, p.user_text]))
        data = await self._chat_json(cfg.rewrite_prompt, source, cfg.rewrite_max_tokens)
        text = str((data or {}).get("text") or "").strip().strip("「」").rstrip("。")
        if not (cfg.min_chars <= len(text) <= cfg.max_chars):
            logger.info("小ネタの書き直しが長さの範囲外でした（%d 字）", len(text))
            return None
        if re.search(cfg.banned_pattern, text) or find_phone(text):
            logger.info("小ネタの書き直しに個人の事情を含みうる語があったので使いません")
            logger.debug("書き直し: %s", text)
            return None
        return text

    def _tidbit_to_offer(self, text: str) -> dict[str, Any] | None:
        """雑談のターンで、この会話でまだ紹介していなければ、ほかの方の小ネタを 1 つ渡す。"""
        if self._tidbits is None or self._session is None or self._session.tidbit_used:
            return None
        if (re.search(self._cfg.question_pattern, text.strip())
                or re.search(self._ref_cfg.procedure_pattern, text)):
            return None  # 尋ねごと・手続きの話には混ぜない
        for item in self._tidbits.active():
            if item.get("id") not in self._session.own_tidbits:
                return item
        return None

    def _known_tidbit(self, text: str) -> bool:
        """もう預かっている小ネタと同じ話か（同じ話を二重に積まない）。"""
        if self._tidbits is None:
            return False
        return any(ngram_overlap(t["text"], text) >= self._tidbit_cfg.used_overlap
                   for t in self._tidbits.active())

    def _tidbit_used(self, tidbit: str, reply: str) -> bool:
        return ("さっき来た方" in reply
                or ngram_overlap(tidbit, reply) >= self._tidbit_cfg.used_overlap)


# 自由な答えを待つ段（はい・いいえで判定しない）と、個人の情報を答える段
_FREE_ANSWER_PHASES = frozenset({"intake", "ask_name", "ask_area", "ask_contact", "ask_phone"})
_PRIVATE_LABELS = {"ask_name": "お名前", "confirm_name": "お名前", "ask_area": "お住まい",
                   "ask_contact": "ご連絡方法", "confirm_contact": "ご連絡方法",
                   "ask_phone": "お電話番号", "confirm_phone": "お電話番号"}
_PAREN_RE = re.compile(r"（(.*?)）")


def _clean_name(text: str, strip_pattern: str, max_chars: int) -> str | None:
    name = re.sub(strip_pattern, "", text.strip())
    name = re.sub(strip_pattern, "", name)  # 「私は山田です。」の前後を両方落とす
    return name if 0 < len(name) <= max_chars else None


def _pick(value: Any, allowed: tuple[str, ...], default: str) -> str:
    return str(value) if value in allowed else default
