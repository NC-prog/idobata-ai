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

from .config import ActionsConfig, ReferralConfig
from .dialogue import ChatBackend, Dialogue
from .knowledge import JST
from .lookup import Lookup
from .referrals import (
    CATEGORIES,
    CONTACT_PREFS,
    URGENCIES,
    ReferralStore,
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


@dataclass
class Session:
    started_at: str
    language: str = "ja"
    turns: int = 0
    backchannel_count: int = 0
    proposals: list[dict[str, str | None]] = field(default_factory=list)
    referral_id: str | None = None


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
        self._words, self._templates = load_templates(cfg.templates_path)
        self._pending: Pending | None = None
        self._session: Session | None = None
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
        self._ensure_session(language)
        assert self._session is not None
        if self._session.turns == 0:
            self._session.language = language
        self._session.turns += 1
        if self._pending is not None and self._is_new_topic(text):
            # 提案には答えず別の話を始めた: 聞き直さずに辞退として、その話に普通に答える
            logger.info("返事待ちに別の話が来たので、提案を辞退扱いにします")
            self._outcome("declined")
            self._pending = None
        if self._pending is not None:
            async for s in self._on_reply(text):
                yield s
            return

        reply: list[str] = []
        async for s in self.dialogue.respond(text, language):
            reply.append(s)
            yield s
        kind = self._adjust_proposal(self.dialogue.last_proposal, text, "".join(reply))
        if kind and self._available(kind):
            line = self._t(kind, "propose")
            self._pending = Pending(kind, text, language)
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
        if (kind == "lookup" and not re.search(self._cfg.question_pattern, text.strip())
                and not re.search(self._cfg.lookup_fallback_topic, text)):
            logger.info("尋ねていない発話への lookup の提案を捨てます")
            return None
        if kind is not None:
            return kind
        if procedure and re.search(self._cfg.refer_fallback_cue, text) and self._available("refer"):
            logger.info("タグはありませんが、手続きの相談なので refer を提案します")
            return "refer"
        if (not procedure and re.search(self._cfg.lookup_fallback_reply, reply)
                and re.search(self._cfg.lookup_fallback_topic, text) and self._available("lookup")):
            logger.info("タグはありませんが、資料に無い行事・営業の質問なので lookup を提案します")
            return "lookup"
        return None

    def _available(self, kind: str) -> bool:
        if kind not in self._cfg.enabled or kind not in self._templates:
            return False
        if kind == "lookup":
            return self._lookup is not None and bool(self._lookup.sources)
        if kind == "refer":
            return self._referrals is not None
        return kind == "share"

    def _t(self, action: str, key: str, **kw: Any) -> str:
        return self._templates.get(action, {}).get(key, "").format(**kw)

    def _outcome(self, outcome: str) -> None:
        if self._session and self._session.proposals:
            self._session.proposals[-1]["outcome"] = outcome

    async def _on_reply(self, text: str) -> AsyncIterator[str]:
        p = self._pending
        assert p is not None
        verdict = classify_reply(text, self._words)
        lines: list[str] = []
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
        fields = await self._structure_referral()
        if fields is None:
            self._pending = None
            yield self._t("refer", "failed")
            return
        p.fields = fields
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
        logger.info("紹介状を保存しました: 受付 %s・%s", ref.receipt, ref.department or "課未定")
        key = "done" if ref.name else "done_no_name"
        yield self._t("refer", key, department=ref.department or "担当の課", number=ref.receipt)

    def _transcript(self, history: list[tuple[str, str]]) -> str:
        return "\n".join(f"来訪者: {u}\n案内役: {a}" for u, a in history)

    async def _structure_referral(self) -> dict[str, Any] | None:
        """会話から紹介状の項目を作る。氏名・電話番号は来訪者が言った場合だけ残す（規則で確かめる）。"""
        history = self.dialogue.history
        user_text = "\n".join(u for u, _ in history)
        departments = [d.name for d in (self._facility.departments if self._facility else ())]
        # 課の名前だけだと取り違える（国保→社会福祉課）ので、担当業務も添える
        dept_lines = [f"{d.name}（{'・'.join(d.handles)}）" if d.handles else d.name
                      for d in (self._facility.departments if self._facility else ())]
        prompt = (self._ref_cfg.structure_prompt
                  + "\nneeds_action は職員の対応が要るか（案内だけで済むなら false）。"
                  + "tags は【タグの一覧】から最大 5 つ選ぶ（一覧に無いものは作らない）。topic_free は話題を 1 行で。"
                  + '\n出力に "needs_action": true, "tags": [], "topic_free": "" も含める。'
                  + "\n【課の一覧】" + "、".join(dept_lines)
                  + "\n【タグの一覧】" + "、".join(self._tags))
        data = await self._chat_json(prompt, self._transcript(history), self._ref_cfg.summary_max_tokens)
        if data is None or not str(data.get("summary") or "").strip():
            return None
        name = data.get("name")
        name = str(name).strip() if isinstance(name, str) and name.strip() and name.strip() in user_text else None
        return {
            "department": str(data.get("department") or "") if data.get("department") in departments else "",
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
        self._pending = None
        yield self._t("share", "failed")

    async def _finish_share(self, p: Pending, ok: bool) -> AsyncIterator[str]:
        self._pending = None
        yield self._t("share", "done" if ok else "declined")


def _pick(value: Any, allowed: tuple[str, ...], default: str) -> str:
    return str(value) if value in allowed else default
