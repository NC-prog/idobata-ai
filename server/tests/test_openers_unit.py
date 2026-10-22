"""WP02 C: 能動的な切り出し・沈黙・別れ（定型文の組み立て・1 会話 1 回・沈黙は 2 回まで）。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from src.camera import PartySnapshot, PartyState
from src.config import SERVER_ROOT, PartyConfig
from src.knowledge import JST, Weather
from src.openers import all_lines, load_openers, opener_lines
from tests.test_camera_unit import _loop

OP = load_openers(SERVER_ROOT / "context" / "openers.yaml")


def test_opener_lines_weather_temperature_and_rotation() -> None:
    assert opener_lines(OP, "こんにちは", "晴れ　後　くもり　夜遅く　雨", 18.0, 0) == [
        "こんにちは。", "今日はいいお天気ですね。", "お元気にされてました？"]  # 予報の最初の天気で選ぶ
    assert opener_lines(OP, "おはようございます", "雨", 7.5, 1) == [
        "おはようございます。", "今日は冷えますね。", "今日はどちらまで？"]  # 冷えるときは気温の一言
    assert opener_lines(OP, "こんばんは", None, None, 2) == ["こんばんは。", "何かお困りのことはありますか？"]
    lines = all_lines(OP, ["おはようございます", "こんにちは", "こんばんは"])
    assert "こんばんは。" in lines and "ゆっくりで大丈夫ですよ。" in lines and "お気をつけて。またどうぞ。" in lines


def test_weather_today(tmp_path: Path) -> None:
    now = datetime(2026, 10, 22, 14, 0, tzinfo=JST)
    path = tmp_path / "weather.json"
    path.write_text(json.dumps({
        "forecasts": [{"date": "2026-10-22", "region": "広島県南部", "weather": "晴れ", "pops": []}],
        "observation": {"observed_at": "2026-10-22T13:40:00+09:00", "temp": 21.4, "place": "江田島",
                        "station": "呉", "in_site": False},
    }, ensure_ascii=False), encoding="utf-8")
    assert Weather(path).today(now) == ("晴れ", 21.4)
    assert Weather(path).today(datetime(2026, 10, 22, 20, 0, tzinfo=JST)) == ("晴れ", None)  # 古い実況は使わない


async def test_opener_once_per_conversation_and_in_history() -> None:
    app, _edge = _loop()
    said: list[tuple[str, list[str]]] = []

    async def fake_respond(text: str, language: str, timing: object, source: Any = None,
                           trigger: str | None = None, speaker: str | None = None) -> None:
        said.append((trigger or "", [s async for s in source()]))

    app._respond_and_speak = fake_respond  # type: ignore[method-assign]
    none, party = PartySnapshot(PartyState.NONE, 0, False), PartySnapshot(PartyState.PARTY, 1, True)
    await app._on_party_change(none, party)
    await asyncio.sleep(0.05)
    assert said[0][0] == "opener" and said[0][1][0] in ("おはようございます。", "こんにちは。", "こんばんは。")
    assert said[0][1][-1] == "お元気にされてました？"
    assert app._dialogue.history[-1][1] == "".join(said[0][1])  # LLM が挨拶を重ねないよう履歴に積む
    await app._on_party_change(party, PartySnapshot(PartyState.PRESENT, 0, True))
    await app._on_party_change(PartySnapshot(PartyState.PRESENT, 0, True), party)  # 同じ会話で輪ができ直した
    await asyncio.sleep(0.05)
    assert [t for t, _ in said] == ["opener"]
    await app._end_conversation()
    await app._on_party_change(none, party)  # 次の会話では問いを回す
    await asyncio.sleep(0.05)
    assert said[-1][1][-1] == "今日はどちらまで？"
    if app._bye_task is not None:
        app._bye_task.cancel()
    await app._tts.close()


async def test_silence_line_after_silence_sec_at_most_twice(monkeypatch: Any) -> None:
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "config", replace(main_mod.config, party=PartyConfig(silence_sec=0.3)))
    monkeypatch.setattr(main_mod, "_SILENCE_POLL_SEC", 0.02)
    app, _edge = _loop()
    assert app._camera is not None
    said: list[str] = []

    async def fake_respond(text: str, language: str, timing: object, source: Any = None,
                           trigger: str | None = None, speaker: str | None = None) -> None:
        said.extend([s async for s in source()])
        app._last_activity = time.monotonic()

    app._respond_and_speak = fake_respond  # type: ignore[method-assign]
    app._camera.push_jpeg(b"\xff\xd8")
    app._camera.tracker.snapshot = PartySnapshot(PartyState.PARTY, 1, True)
    app._opener_said = True
    app._last_activity = time.monotonic()
    task = asyncio.create_task(app._watch_silence())
    await asyncio.sleep(0.2)
    assert said == []  # まだ黙って 0.3 秒経っていない
    for _ in range(40):
        app._camera.push_jpeg(b"\xff\xd8")  # カメラは動き続けている
        await asyncio.sleep(0.05)
    task.cancel()
    assert said == ["ゆっくりで大丈夫ですよ。", "ほかにも何かあれば、どうぞ。"]  # 1 会話 2 回まで
    await app._tts.close()
