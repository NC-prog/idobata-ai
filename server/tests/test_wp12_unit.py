"""WP12: 来訪者の発話区間の音声（保存・掃除・保全・管理者ページでの再生・素材への書き出し）。"""

from __future__ import annotations

import asyncio
import json
import tempfile
import wave
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from src.config import AdminConfig, AudioRetentionConfig
from src.knowledge import JST
from src.transcripts import AudioStore, TranscriptStore, new_utterance_id
from tests.test_main_unit import _loop

DAY = date(2026, 10, 24)


def _store(tmp_path: Path, days: int = 7) -> tuple[TranscriptStore, AudioStore]:
    t = TranscriptStore(tmp_path, "etajima", 7)
    return t, AudioStore(t, AudioRetentionConfig(retention_days=days, min_free_bytes=0))


def _wave(sec: float = 0.5) -> np.ndarray:
    n = int(16000 * sec)
    return (0.3 * np.sin(np.linspace(0, 440 * 2 * np.pi * sec, n))).astype(np.float32)


def test_save_writes_16k_mono_16bit_and_paths(tmp_path: Path) -> None:
    _, a = _store(tmp_path)
    uid = new_utterance_id()
    assert a.save(DAY.isoformat(), uid, _wave())
    path = a.file(DAY.isoformat(), uid)
    assert path is not None and path == tmp_path / "transcripts" / "etajima" / "2026-10-24" / "audio" / f"{uid}.wav"
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 8000)
    assert a.relpath("2026-10-24", uid) == f"2026-10-24/audio/{uid}.wav"
    assert a.day_size("2026-10-24") == (1, path.stat().st_size)
    # 形の違う日付・id は配信しない
    assert a.file("../2026-10-24", uid) is None and a.file("2026-10-24", "../x") is None


def test_retention_zero_saves_nothing(tmp_path: Path) -> None:
    _, a = _store(tmp_path, days=0)
    assert not a.enabled and not a.save(DAY.isoformat(), new_utterance_id(), _wave())
    assert not (tmp_path / "transcripts").exists()


def test_low_disk_stops_saving_and_warns_once(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    import shutil

    t = TranscriptStore(tmp_path, "etajima", 7)
    a = AudioStore(t, AudioRetentionConfig(min_free_bytes=10 ** 15))
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr("src.transcripts.shutil.disk_usage", lambda p: usage)
    assert not a.save("2026-10-24", new_utterance_id(), _wave())
    assert not a.save("2026-10-24", new_utterance_id(), _wave())
    assert sum("ディスクの空き" in r.message for r in caplog.records) == 1


def test_cleanup_removes_old_audio_but_keeps_held(tmp_path: Path) -> None:
    t, a = _store(tmp_path, days=7)
    old = (DAY - timedelta(days=10)).isoformat()
    for sid, uid in (("held1", "100000-aaaaaa"), ("free1", "100001-bbbbbb")):
        a.save(old, uid, _wave(0.1))
        t.root.mkdir(parents=True, exist_ok=True)
        with (t.root / f"{old}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": f"{old}T10:00:00+09:00", "session_id": sid, "speaker": "user", "id": uid,
                                "audio": a.relpath(old, uid)}) + "\n")
    a.save(DAY.isoformat(), "100002-cccccc", _wave(0.1))
    assert t.set_hold("held1", True)
    assert a.cleanup(DAY) == 1
    assert a.file(old, "100000-aaaaaa") is not None  # 保全したセッションの音声は残る
    assert a.file(old, "100001-bbbbbb") is None
    assert a.file(DAY.isoformat(), "100002-cccccc") is not None  # 期限内
    # 保全を外すと次の掃除で消え、空になった日のディレクトリも消える
    t.set_hold("held1", False)
    assert a.cleanup(DAY) == 1 and not (t.root / old).exists()


async def test_admin_audio_route_range_and_no_store(tmp_path: Path) -> None:
    from src.edge_server import AdminPanel, create_app

    t, a = _store(tmp_path)
    uid = "101010-abcdef"
    day = datetime.now(JST).date().isoformat()
    a.save(day, uid, _wave())
    t.append("user", "s1", "edge", id=uid, text="住民票はどこ", audio=a.relpath(day, uid), asr_status="ok")
    t.append("user", "s2", "edge", id="101011-abcdef", text="こんにちは")  # 音声の無い行（WP12 より前の形）
    app = create_app(tmp_path / "none", admin=AdminPanel(t, AdminConfig(password="pw"), a))
    transport = httpx.ASGITransport(app=app, client=("192.168.1.20", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        auth = ("admin", "pw")
        url = f"/admin/transcripts/audio/{day}/{uid}"
        assert (await c.get(url)).status_code == 401
        full = await c.get(url, auth=auth)
        assert full.status_code == 200 and full.headers["cache-control"] == "no-store"
        part = await c.get(url, auth=auth, headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.content) == 100
        assert (await c.get(f"/admin/transcripts/audio/{day}/nothere-000000", auth=auth)).status_code == 404
        sessions = (await c.get(f"/admin/api/transcripts?day={day}", auth=auth)).json()
        assert {s["session_id"]: s["has_audio"] for s in sessions} == {"s1": True, "s2": False}
        size = (await c.get(f"/admin/api/transcripts/audio_size?day={day}", auth=auth)).json()
        assert size["files"] == 1 and size["bytes"] > 16000
        page = (await c.get("/admin/transcripts/", auth=auth)).text
        assert "音声" in page and "__AUDIO_DAYS__" not in page


class _ASR:
    def __init__(self, text: str, language: str = "ja") -> None:
        self.text, self.language = text, language

    async def transcribe_segment(self, frames: Any, t: float, **kw: Any) -> Any:
        class C:
            is_final = True

        c = C()
        c.text, c.language = self.text, self.language  # type: ignore[attr-defined]
        yield c

    def push_rolling_context(self, text: str) -> None:
        pass


def _app(tmp_path: Path, asr_text: str, days: int = 7) -> Any:
    app, _rec = _loop()
    app._transcripts = TranscriptStore(tmp_path, "etajima", 7)
    app._audio = AudioStore(app._transcripts, AudioRetentionConfig(retention_days=days, min_free_bytes=0))
    app._asr = _ASR(asr_text)  # type: ignore[assignment]

    async def no_reply(*a: Any, **kw: Any) -> None:
        return None

    app._respond_and_speak = no_reply  # type: ignore[method-assign]
    return app


async def _settle(app: Any) -> None:
    for _ in range(50):
        if not [t for t in app._bg_tasks if t.get_name() == "audio-save"]:
            return
        await asyncio.sleep(0.02)


async def test_pipeline_saves_audio_with_transcript_row(tmp_path: Path) -> None:
    app = _app(tmp_path, "住民票はどこで取れますか")
    frames = [_wave(0.032) for _ in range(30)]
    await app._on_segment(frames, 0.0)
    await _settle(app)
    (row,) = [r for r in app._transcripts.read(datetime.now(JST).date()) if r["speaker"] == "user"]
    assert row["asr_status"] == "ok" and row["audio"].endswith(f"{row['id']}.wav")
    assert (app._transcripts.root / row["audio"]).is_file()
    await app._tts.close()


async def test_pipeline_saves_unheard_and_short_but_not_outside_party(tmp_path: Path) -> None:
    app = _app(tmp_path, "")  # ASR が何も返さない
    await app._on_segment([_wave(0.032) for _ in range(30)], 0.0)
    app._on_short_segment([_wave(0.032) for _ in range(3)])
    app._party_at_start = False
    app._camera_listening = lambda: False  # type: ignore[method-assign]
    app._on_short_segment([_wave(0.032) for _ in range(3)])  # 輪が無い: 残さない
    await app._on_segment([_wave(0.032) for _ in range(30)], 0.0)  # 輪が無い: ASR にも渡さない・残さない
    await _settle(app)
    rows = [r for r in app._transcripts.read(datetime.now(JST).date()) if r["speaker"] == "user"]
    assert [r["asr_status"] for r in rows] == ["empty", "rejected_short"]
    assert all((app._transcripts.root / r["audio"]).is_file() for r in rows)
    assert len(list((app._transcripts.root / datetime.now(JST).date().isoformat() / "audio").iterdir())) == 2
    await app._tts.close()


async def test_pipeline_saves_nothing_when_disabled(tmp_path: Path) -> None:
    app = _app(tmp_path, "住民票はどこで取れますか", days=0)
    await app._on_segment([_wave(0.032) for _ in range(30)], 0.0)
    await _settle(app)
    (row,) = [r for r in app._transcripts.read(datetime.now(JST).date()) if r["speaker"] == "user"]
    assert "audio" not in row and row["asr_status"] == "ok" and row["id"]
    assert not (app._transcripts.root / datetime.now(JST).date().isoformat()).exists()
    await app._tts.close()


async def test_keep_audio_does_not_block_the_turn(tmp_path: Path) -> None:
    import time

    app = _app(tmp_path, "x")
    frames = [_wave(0.032) for _ in range(300)]  # 約 10 秒
    app._keep_audio("115959-abcdef", frames)  # 空打ち（初回だけのスレッドの用意を測らない）
    took = []
    for i in range(3):  # 3 回のうち最も速い値（全体を流したときの GC などの一瞬の止まりを測らない）
        t0 = time.perf_counter()
        rel = app._keep_audio(f"12000{i}-abcdef", frames)
        took.append((time.perf_counter() - t0) * 1000)
    assert rel is not None and min(took) < 50, took  # 書き込みは別スレッド（ターンの遅れは 50ms 未満）
    await _settle(app)
    assert (app._transcripts.root / rel).is_file()
    await app._tts.close()


def test_export_review_copies_audio_only_with_flag(tmp_path: Path) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("export_review", Path(__file__).resolve().parents[2] / "scripts" /
                                                  "export_review.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_review"] = mod
    spec.loader.exec_module(mod)
    t, a = _store(tmp_path)
    a.save("2026-10-24", "130000-abcdef", _wave(0.1))
    rows = [{"ts": "2026-10-24T13:00:00+09:00", "session_id": "s", "speaker": "user", "id": "130000-abcdef",
             "text": "", "asr_status": "empty", "audio": "2026-10-24/audio/130000-abcdef.wav"}]
    items = mod.extract(rows, [])
    assert items == [{"session_id": "s", "kind": "unheard", "ts": rows[0]["ts"], "utterance_id": "130000-abcdef",
                      "asr_status": "empty"}]
    out = Path(tempfile.mkdtemp()) / "a"
    assert mod.copy_audio(rows, items, t.root, out) == 1 and (out / "130000-abcdef.wav").is_file()


async def test_privacy_answer_follows_audio_setting(monkeypatch: Any) -> None:
    """「録音してる？」には、音声を残す設定なら文字と声の記録、残さない設定なら「声は録音していません」（掲示文言どおり）。"""
    from dataclasses import replace

    import src.dialogue as dialogue_mod
    from src.config import DialogueConfig, config
    from src.dialogue import Dialogue
    from tests.test_dialogue_unit import StubLLM, _collect

    d = Dialogue(StubLLM([]), DialogueConfig())
    out = await _collect(d, "今の話、録音してる？")
    assert "文字と声の記録" in out[0] and "外に出ることはありません" in out[1]
    monkeypatch.setattr(dialogue_mod, "config", replace(config, audio=AudioRetentionConfig(retention_days=0)))
    out = await _collect(Dialogue(StubLLM([]), DialogueConfig()), "今の話、録音してる？")
    assert out[0] == "声は録音していません。"
