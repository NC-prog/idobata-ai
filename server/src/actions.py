"""同意を取ってから外に働きかける（WP04 A-2 / A-3 / B-1）。

1 会話の流れ（Conversation.respond）:

- LISTEN: 通常の対話。LLM の応答の最後に <propose:lookup|refer|share> があれば、行為ごとの
  定型文で提案して PROPOSED へ（提案の言葉は LLM に作らせない）
- PROPOSED: 来訪者の発話は通常の対話に流さず、返事だけを規則で判定する（同意 / 辞退 / 不明）。
  不明なら 1 回だけ聞き直し、それでも不明なら辞退扱い
- 同意 → 「少々お待ちくださいね」→ 実行（エッジは thinking）→ 結果を 1〜3 文で報告 → LISTEN
- 紹介状で電話番号を言っていたら、復唱して確かめる（CONFIRM_PHONE）

会話の区切り（待機タイムアウト）で、匿名の来訪記録を 1 行書く（end_session）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .asr_client import ngram_overlap
from .config import ActionsConfig, ReferralConfig, TidbitConfig
from .dialogue import ChatBackend, Dialogue
from .knowledge import JST
from .lookup import Lookup
from .procedures import Procedures
from .referrals import (
    CATEGORIES,
    CONTACT_PREFS,
    UNSPECIFIED_DEPARTMENT,
    URGENCIES,
    ReferralStore,
    TidbitStore,
    VisitLog,
    find_phone,
    phone_for_speech,
    pick_tags,
)
from .site_context import Facility

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ReplyWords:
    strong_yes: list[str]
    yes_override: list[str]
    no: list[str]
    yes: list[str]


def load_templates(path: Path) -> tuple[ReplyWords, dict[str, dict[str, str]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    r = data.get("reply") or {}
    words = ReplyWords(*([str(x) for x in r.get(k) or []]
                         for k in ("strong_consent", "consent_override", "decline", "consent")))
    actions = {str(k): {str(a): str(b) for a, b in (v or {}).items()} for k, v in (data.get("actions") or {}).items()}
    return words, actions


def classify_reply(text: str, words: ReplyWords) -> str:
    """返事の判定（"yes" / "no" / "unclear"）。LLM は使わない。

    はっきりした同意（お願い・頼む）→ 辞退の語を含む同意（いいですよ）→ 辞退 → 同意、の順に見る。
    「はい、大丈夫です」は辞退（よくある言い方）、「いや、お願いします」は同意。
    """
    t = text.strip()
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
    phase: str = "proposed"  # proposed / confirm_phone / confirm_share
    reasks: int = 0
    fields: dict[str, Any] | None = None  # 紹介状の中身（電話番号の確認待ち）
    share_text: str | None = None
    # 提案したターンの番号と、そのときの話題の始まり（紹介状の要約に使う範囲）
    turn_index: int = 0
    topic_floor: int = 0
    # 提案の後の来訪者の返事（「お願い。電話は…」など）
    replies: list[str] = field(default_factory=list)


@dataclass
class Session:
    started_at: str
    language: str = "ja"
    turns: int = 0
    backchannel_count: int = 0
    proposals: list[dict[str, str | None]] = field(default_factory=list)
    referral_id: str | None = None
    # 小ネタ: この会話で紹介したか（1 会話 1 つまで）、この会話で預かったもの（本人には返さない）
    tidbit_used: bool = False
    own_tidbits: list[str] = field(default_factory=list)


StateHook = Callable[[str], Awaitable[None]]


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
        # この会話のターン（来訪者, 応答）と、いまの話題が始まったターンの番号（プロセス内のみ）
        self._turns: list[tuple[str, str]] = []
        self._topic_start = 0
        self._turn_new_topic = False
        self._words, self._templates = load_templates(cfg.templates_path)
        self._pending: Pending | None = None
        self._session: Session | None = None
        # 直前のターンで行為が終わったか（次の短い相づちを決まった文で受ける）
        self._after_action = False
        # 直近の紹介状の受付番号（字幕・表示用）
        self.last_receipt: str | None = None

    # ------------------------------------------------------------------
    # 会話の区切り
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        return "listen" if self._pending is None else self._pending.phase

    @property
    def language(self) -> str:
        """返事待ちの行為の言語（無ければ会話の言語、まだ無ければ ja）。"""
        if self._pending is not None:
            return self._pending.language
        return self._session.language if self._session is not None else "ja"

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
        session, history = self._session, self.dialogue.history
        self._session, self._pending = None, None
        self._turns, self._topic_start = [], 0
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
        }
        try:
            self._visits.write(record)
        except OSError:
            logger.warning("来訪記録を書けません", exc_info=True)

    # ------------------------------------------------------------------
    # 1 ターン
    # ------------------------------------------------------------------

    async def respond(self, text: str, language: str = "ja") -> AsyncIterator[str]:
        """1 発話に応答する。ターンを記録し、行為が終わったら話題の区切りとする。"""
        index, had_pending = len(self._turns), self._pending is not None
        self._turn_new_topic = False
        lines: list[str] = []
        async for s in self._respond_turn(text, language, index):
            lines.append(s)
            yield s
        self._turns.append((text, "".join(lines)))
        if had_pending and self._pending is None and not self._turn_new_topic:
            self._topic_start = len(self._turns)  # 行為が終わった・断られた: 次のターンから別の話題

    async def _respond_turn(self, text: str, language: str, index: int) -> AsyncIterator[str]:
        self._ensure_session(language)
        assert self._session is not None
        if self._session.turns == 0:
            self._session.language = language
        self._session.turns += 1
        after_action, self._after_action = self._after_action, False
        if (after_action and self._pending is None and classify_reply(text, self._words) != "unclear"
                and len(re.sub(r"[\s、。！？!?]", "", text)) <= self._cfg.reply_max_chars):
            # 行為が終わった直後の「うん」「ありがとう」は決まった文で受ける
            line = self._t("common", "ack")
            yield line
            self.dialogue.add_turn(text, line)
            return
        if self._pending is not None and self._is_new_topic(text):
            # 提案には答えず別の話を始めた: 聞き直さずに辞退として、その話に普通に答える
            logger.info("返事待ちに別の話が来たので、提案を辞退扱いにします")
            self._outcome("declined")
            self._pending = None
            self._topic_start, self._turn_new_topic = index, True
        if self._pending is not None:
            async for s in self._on_reply(text):
                yield s
            self._after_action = self._pending is None
            return

        reply: list[str] = []
        offered = self._tidbit_to_offer(text)
        hint = self._tidbit_cfg.offer_template.format(text=offered["text"]) if offered else ""
        async for s in self.dialogue.respond(text, language, hint=hint):
            reply.append(s)
            yield s
        if offered and self._session and self._tidbit_used(offered["text"], "".join(reply)):
            self._session.tidbit_used = True
            logger.info("さっきの方の小ネタを紹介しました")
        kind = self._adjust_proposal(self.dialogue.last_proposal, text, "".join(reply))
        if kind and self._available(kind):
            line = self._t(kind, "propose")
            self._pending = Pending(kind, text, language, turn_index=index, topic_floor=self._topic_start)
            self._session.proposals.append({"action": kind, "outcome": None})
            self.dialogue.extend_last_reply(line)
            logger.info("行為を提案しました: %s", kind)
            yield line

    def _is_new_topic(self, text: str) -> bool:
        p = self._pending
        return (p is not None and p.phase == "proposed"
                and classify_reply(text, self._words) == "unclear"
                and len(re.sub(r"[\s、。！？!?]", "", text)) > self._cfg.reply_max_chars)

    def _adjust_proposal(self, kind: str | None, text: str, reply: str) -> str | None:
        """LLM のタグを規則で補正する（手続きの相談は lookup より refer、付け忘れの補い）。"""
        procedure = re.search(self._ref_cfg.procedure_pattern, text) is not None
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

    def _share_fits(self, text: str) -> bool:
        """覚えておくの提案に合う雑談か（尋ねていない・手続きでない・個人の事情の語が無い・1 会話 1 回）。"""
        if not self._available("share") or self._session is None:
            return False
        if any(p.get("action") == "share" for p in self._session.proposals):
            return False
        return (re.search(self._cfg.share_fallback_topic, text) is not None
                and not self._known_tidbit(text)
                and not re.search(self._cfg.question_pattern, text.strip())
                and not re.search(self._ref_cfg.procedure_pattern, text)
                and not re.search(self._tidbit_cfg.banned_pattern, text))

    def _available(self, kind: str) -> bool:
        if kind not in self._cfg.enabled or kind not in self._templates:
            return False
        if kind == "lookup":
            return self._lookup is not None and bool(self._lookup.sources)
        if kind == "refer":
            return self._referrals is not None
        return kind == "share" and self._tidbits is not None

    def _t(self, action: str, key: str, **kw: Any) -> str:
        return self._templates.get(action, {}).get(key, "").format(**kw)

    def _outcome(self, outcome: str) -> None:
        if self._session and self._session.proposals:
            self._session.proposals[-1]["outcome"] = outcome

    async def _on_reply(self, text: str) -> AsyncIterator[str]:
        p = self._pending
        assert p is not None
        p.replies.append(text)
        lines: list[str] = []
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
            async for s in self._act(p):
                lines.append(s)
                yield s
        elif p.phase == "confirm_phone":
            assert p.fields is not None
            if verdict == "no":
                p.fields["phone"] = None
                line = self._t("refer", "phone_wrong")
                lines.append(line)
                yield line
            async for s in self._save_referral(p):
                lines.append(s)
                yield s
        elif p.phase == "confirm_share":
            async for s in self._finish_share(p, verdict == "yes"):
                lines.append(s)
                yield s
        self.dialogue.add_turn(text, "".join(lines))

    def _reask_line(self, p: Pending) -> str:
        if p.phase == "confirm_phone":
            return self._t("refer", "confirm_phone", phone=phone_for_speech((p.fields or {}).get("phone") or ""))
        if p.phase == "confirm_share":
            return self._t("share", "confirm_text", text=p.share_text or "")
        return self._t(p.action, "reask")

    async def _act(self, p: Pending) -> AsyncIterator[str]:
        if p.action in ("lookup", "refer"):
            yield self._t(p.action, "acting")
            if self._on_state is not None:
                await self._on_state("thinking")
        elif self._on_state is not None:
            await self._on_state("thinking")  # 書き直しの間（言葉は挟まない。すぐ確認の文が出る）
        if p.action == "lookup":
            self._pending = None
            async for s in self._do_lookup(p):
                yield s
        elif p.action == "refer":
            async for s in self._do_refer(p):
                yield s
        elif p.action == "share":
            async for s in self._do_share(p):
                yield s

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
        async for s in self.dialogue.respond(p.user_text, p.language, extra_refs=result.references):
            yield s

    # ---- つなぐ（紹介状）------------------------------------------------

    async def _do_refer(self, p: Pending) -> AsyncIterator[str]:
        window = self._refer_window(p)
        fields = await self._structure_referral(window)
        if fields is None:
            self._pending = None
            yield self._t("refer", "failed")
            return
        p.fields = fields
        # 担当課は規則だけで決める（直近の発話 = 提案のきっかけとその後の返事。決まらなければ話題全体）。
        # 一意に決まらなければ「未特定」で保存し、職員が職員パネルで課を付ける。来訪者には聞き返さない
        recent = "\n".join([p.user_text, *p.replies])
        cands = (self._procedures.departments_for(recent, self._facility)
                 or self._procedures.departments_for("\n".join(u for u, _ in window), self._facility))
        if len(cands) == 1:
            fields["department"] = cands[0]
        else:
            fields["department"] = UNSPECIFIED_DEPARTMENT
            logger.info("担当課が規則で一意に決まりません（候補 %d）。未特定で保存します", len(cands))
        if fields.get("phone"):
            p.phase, p.reasks = "confirm_phone", 0
            yield self._t("refer", "confirm_phone", phone=phone_for_speech(fields["phone"]))
            return
        async for s in self._save_referral(p):
            yield s

    async def _save_referral(self, p: Pending) -> AsyncIterator[str]:
        assert self._referrals is not None and p.fields is not None
        self._pending = None
        try:
            ref = await asyncio.to_thread(self._referrals.create, p.fields, self._edge_id(), p.language)
        except Exception:
            logger.exception("紹介状を保存できません")
            yield self._t("refer", "failed")
            return
        self.last_receipt = ref.receipt
        if self._session is not None:
            self._session.referral_id = ref.id
        logger.info("紹介状を保存しました: 受付 %s・%s", ref.receipt, ref.department)
        unknown = ref.department in ("", UNSPECIFIED_DEPARTMENT)
        key = ("done_unknown" if unknown else "done") + ("" if ref.name else "_no_name")
        yield self._t("refer", key, department=ref.department, number=ref.receipt)

    def _transcript(self, history: list[tuple[str, str]]) -> str:
        return "\n".join(f"来訪者: {u}\n案内役: {a}" for u, a in history)

    def _refer_window(self, p: Pending) -> list[tuple[str, str]]:
        """紹介状を作る範囲: 提案のきっかけの発話と、その前後 2 往復（いまの話題の中だけ）。"""
        start = max(p.turn_index - self._ref_cfg.window_turns, p.topic_floor)
        before = self._turns[start:p.turn_index + 1]
        after = [(r, "") for r in p.replies[: self._ref_cfg.window_turns]]
        return before + after

    async def _structure_referral(self, history: list[tuple[str, str]]) -> dict[str, Any] | None:
        """会話から紹介状の項目を作る。氏名・電話番号は来訪者が言った場合だけ残す（規則で確かめる）。

        課は LLM に決めさせない（規則で決める。_do_refer）。
        """
        user_text = "\n".join(u for u, _ in history)
        prompt = (self._ref_cfg.structure_prompt
                  + "\nneeds_action は職員の対応が要るか（案内だけで済むなら false）。"
                  + "tags は【タグの一覧】から最大 5 つ選ぶ（一覧に無いものは作らない）。topic_free は話題を 1 行で。"
                  + '\n出力に "needs_action": true, "tags": [], "topic_free": "" も含める。'
                  + "\n【タグの一覧】" + "、".join(self._tags))
        data = await self._chat_json(prompt, self._transcript(history), self._ref_cfg.summary_max_tokens)
        if data is None or not str(data.get("summary") or "").strip():
            return None
        name = data.get("name")
        name = str(name).strip() if isinstance(name, str) and name.strip() and name.strip() in user_text else None
        return {
            "department": "",
            "category": _pick(data.get("category"), CATEGORIES, "その他"),
            "summary": str(data.get("summary")).strip(),
            "urgency": _pick(data.get("urgency"), URGENCIES, "通常"),
            "contact_pref": _pick(data.get("contact_pref"), CONTACT_PREFS, "未定"),
            "name": name,
            # 電話番号は LLM の出力ではなく、来訪者の発話から規則で拾う（言っていなければ無し）
            "phone": find_phone(user_text),
            "needs_action": bool(data.get("needs_action", True)),
            "tags": pick_tags(data.get("tags"), self._tags),
            "topic_free": str(data.get("topic_free") or "")[:60],
        }

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


def _pick(value: Any, allowed: tuple[str, ...], default: str) -> str:
    return str(value) if value in allowed else default
