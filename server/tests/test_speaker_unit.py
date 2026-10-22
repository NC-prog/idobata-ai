"""WP02 B: セッション限りの話者の区別（まとめ方・受け付けの規則・LLM へのラベル・付き添いの答え・来訪記録）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.camera import PartyState
from src.config import DialogueConfig, SpeakerConfig
from src.dialogue import Dialogue
from src.referrals import VisitLog
from tests.test_actions_unit import PHRASED, SpyLLM, _conv, _refer_json, _saved
from tests.test_dialogue_unit import StubLLM

CFG = SpeakerConfig()


def _v(*xs: float) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[: len(xs)] = xs
    return v


def test_clusters_label_and_accept_by_party_state() -> None:
    from src.speaker import SpeakerClusters

    c = SpeakerClusters(CFG)
    a = c.assign(_v(1, 0.1), 2.0, PartyState.PARTY)
    b = c.assign(_v(0.1, 1), 2.0, PartyState.PARTY)
    assert (a.label, a.accepted, a.new) == ("A", True, True)
    assert (b.label, b.accepted) == ("B", True)
    assert c.assign(_v(1, 0.2), 1.5, PartyState.PARTY).label == "A"  # 同じ人
    assert c.assign(_v(0.5, 0.5), 0.4, PartyState.PARTY).label in ("A", "B")  # 短い返事は新しい人にしない
    # 輪が無いときに初めて聞いた声は、後で輪の中で話しても相手にしない
    bg = c.assign(_v(0, 0, 1), 2.0, PartyState.NONE)
    assert (bg.label, bg.accepted) == ("C", False)
    assert not c.assign(_v(0, 0.1, 1), 2.0, PartyState.PARTY).accepted
    # 近づいてくる途中（PRESENT）に初めて聞いた声は保留、輪ができてから話せば相手
    p = c.assign(_v(0, 0, 0, 1), 2.0, PartyState.PRESENT)
    assert not p.accepted
    assert c.assign(_v(0, 0, 0, 1), 2.0, PartyState.PARTY).accepted
    assert c.accepted_count == 3
    c.reset()
    assert c.accepted_count == 0 and c.assign(_v(1), 2.0, PartyState.PARTY).label == "A"  # 会話ごとに捨てる


def test_clusters_are_capped() -> None:
    from dataclasses import replace

    from src.speaker import SpeakerClusters

    c = SpeakerClusters(replace(CFG, max_clusters=3))
    for i in range(5):
        c.assign(np.eye(8, dtype=np.float32)[i], 2.0, PartyState.NONE)
    assert len(c._clusters) == 3


async def test_dialogue_gets_labeled_utterances_and_label_is_not_spoken() -> None:
    llm = StubLLM(["A：はい、市民課です。"])
    d = Dialogue(llm, DialogueConfig())
    out = [s async for s in d.respond("住民票はどこ", "ja", speaker="B")]
    assert "".join(out) == "はい、市民課です。"  # 頭の「A：」は読み上げない
    messages = llm.calls[-1][1]
    assert messages[-1]["content"].endswith("B：住民票はどこ")
    assert d.history[-1][0] == "B：住民票はどこ"
    assert "声で分けた記号" in messages[0]["content"]


async def test_companion_answers_are_marked_and_visit_counts_speakers(tmp_path: Path) -> None:
    llm = SpyLLM([["国保ですね。<propose:refer>"]], chat_reply=[PHRASED, _refer_json(), '{"tags": []}'])
    c = _conv(llm, tmp_path)
    c._speaker_count = lambda: 2
    c._visits = VisitLog(tmp_path, "etajima")
    await _say_as(c, "国保の切り替えで聞きたいことがあって", "A")
    await _say_as(c, "お願いします", "A")
    await _say_as(c, "会社を退職したので", "B")  # 付き添いが代わりに答える
    await _say_as(c, "来月からです", "A")
    await c.end_session()
    ref = _saved(tmp_path)
    assert [q["answer"] for q in ref.intake][:2] == ["会社を退職したので（付き添いの方の答え）", "来月からです"]
    from datetime import datetime

    from src.knowledge import JST

    (visit,) = VisitLog(tmp_path, "etajima").read(datetime.now(JST).date())
    assert visit["speakers"] == 2


async def _say_as(c: Any, text: str, speaker: str) -> list[str]:
    return [s async for s in c.respond(text, "ja", speaker)]
