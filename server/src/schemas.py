"""議事録データ構造の定義。

タウンホールミーティングの文字起こし・要約・ToDo抽出で使用する
Pydantic モデルを集約する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class TranscriptChunk:
    """ASR から返される文字起こしチャンク。レイテンシ計測用タイムスタンプ付き。"""

    text: str
    is_final: bool
    t_speech_start: float
    t_emitted: float
    # ASR が検出した言語（ISO 639-1。例: "ja", "en", "vi"）
    language: str = "ja"

    @property
    def latency_sec(self) -> float:
        """発話開始からチャンク発行までのレイテンシ（秒）。"""
        return self.t_emitted - self.t_speech_start


class TranscriptSegment(BaseModel):
    """文字起こしの1セグメント。"""

    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    presenter_id: str | None = None


class ActionItem(BaseModel):
    """アクションアイテム（ToDo）。"""

    description: str
    assignee: str | None = None
    due: str | None = None
    source_presentation: str = ""


class PresentationSummary(BaseModel):
    """発表ごとの要約。"""

    presenter: str
    topic: str
    key_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class MeetingMinutes(BaseModel):
    """タウンホール全体の議事録。"""

    title: str
    date: str
    presentations: list[PresentationSummary] = Field(default_factory=list)
    all_action_items: list[ActionItem] = Field(default_factory=list)
    closing_note: str = ""
    quiz_answers: list[str] = Field(default_factory=list)
