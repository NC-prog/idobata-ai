"""WP00.5 TTS 聞き比べ: VOICEVOX vs Qwen3-TTS 0.6B の計測スクリプト。

アプリ本体（server/）とは独立。評価用の合成音声だけを out/ に保存する。

使い方:
    uv run python compare.py            # 全計測 + results.csv + index.html
    uv run python compare.py --runs 1   # 試運転

計測の定義（docs/WP00.5_tts_comparison.md 1-3）:
- VOICEVOX: audio_query + synthesis 完了 = 初回音声 = 全音声（ストリーミング不可）
- Qwen3-TTS: 公式 qwen-tts パッケージにはストリーミング出力 API が無いため、
  talker（コーデック生成器）の forward フックで生成中のコーデックフレームを受け取り、
  先頭 FIRST_CHUNK_FRAMES フレームが揃った時点でそれだけをデコードして
  「最初の音声チャンク」とみなす。全音声は generate_custom_voice() の戻りまで。
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import logging
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger("compare")

LAB_DIR = Path(__file__).resolve().parent
OUT_DIR = LAB_DIR / "out"

# --- VOICEVOX（server/src/config.py の VoicevoxConfig と同じ値に合わせる） ---
VOICEVOX_URL = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER = 2
VOICEVOX_QUERY_OVERRIDES = {
    "speedScale": 1.0,
    "pitchScale": 0.0,
    "intonationScale": 1.0,
    "volumeScale": 1.0,
}
VOICEVOX_TIMEOUT_SEC = 30.0
VOICEVOX_RETRIES = 2

# --- Qwen3-TTS ---
QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN_SPEAKER = "Ono_Anna"  # README: 日本語ネイティブのプリセット話者は Ono_Anna のみ
QWEN_LANG = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
# 12Hz トークナイザ: 1 フレーム ≒ 83ms。4 フレーム ≒ 0.33 秒を初回チャンクとする
FIRST_CHUNK_FRAMES = 4

VRAM_POLL_SEC = 0.1


@dataclass
class Sentence:
    lang: str
    n: int
    text: str


@dataclass
class Measure:
    t_first_ms: float
    t_total_ms: float
    audio: np.ndarray
    sr: int
    vram_peak_mb: float | None = None


@dataclass
class Row:
    engine: str
    lang: str
    n: int
    chars: int
    t_first_audio_ms: float
    t_total_ms: float
    audio_sec: float
    rtf: float
    vram_peak_mb: float | None
    runs: list[tuple[float, float]] = field(default_factory=list)


def load_sentences() -> list[Sentence]:
    out: list[Sentence] = []
    ja = (LAB_DIR / "sentences_ja.txt").read_text(encoding="utf-8").splitlines()
    out += [Sentence("ja", i + 1, t.strip()) for i, t in enumerate(ja) if t.strip()]
    for lang in ("en", "zh"):
        for line in (LAB_DIR / f"sentences_{lang}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                n, text = line.split("\t", 1)
                out.append(Sentence(lang, int(n), text.strip()))
    return out


def gpu_used_mb() -> float:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return float(r.stdout.strip().splitlines()[0])


class VramPeak:
    """合成中に nvidia-smi をポーリングして GPU 全体の使用量ピークを取る。"""

    def __init__(self) -> None:
        self.peak = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, gpu_used_mb())
            except Exception:
                logger.debug("nvidia-smi 失敗", exc_info=True)
            self._stop.wait(VRAM_POLL_SEC)

    def __enter__(self) -> Self:
        self._th.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._th.join()


class VoicevoxEngine:
    name = "voicevox"

    def __init__(self) -> None:
        self.client = httpx.Client(base_url=VOICEVOX_URL, timeout=VOICEVOX_TIMEOUT_SEC)

    def _post(self, path: str, **kw: Any) -> httpx.Response:
        for attempt in range(VOICEVOX_RETRIES + 1):
            try:
                r = self.client.post(path, **kw)
                r.raise_for_status()
                return r
            except httpx.HTTPError:
                if attempt == VOICEVOX_RETRIES:
                    raise
                logger.warning("VOICEVOX %s 失敗、リトライ %d", path, attempt + 1)
                time.sleep(0.5)
        raise AssertionError("unreachable")

    def synth(self, s: Sentence) -> Measure:
        t0 = time.perf_counter()
        q = self._post("/audio_query", params={"text": s.text, "speaker": VOICEVOX_SPEAKER}).json()
        q.update(VOICEVOX_QUERY_OVERRIDES)
        wav = self._post("/synthesis", params={"speaker": VOICEVOX_SPEAKER}, json=q).content
        t1 = time.perf_counter()
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        ms = (t1 - t0) * 1000
        return Measure(ms, ms, audio, sr)


class QwenEngine:
    name = "qwen3tts-0.6b"

    def __init__(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(
            QWEN_MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        self._frames: list[Any] = []
        self._t_first: float | None = None
        self._first_audio: np.ndarray | None = None
        self.model.model.talker.register_forward_hook(self._on_step)

    def _on_step(self, _mod: Any, _inp: Any, out: Any) -> None:
        """talker の 1 ステップごとに呼ばれる。先頭チャンクが揃ったらデコードして時刻を記録。"""
        codec_ids = out.hidden_states[1]
        if codec_ids is None or self._t_first is not None:
            return
        self._frames.append(codec_ids[0].detach())
        if len(self._frames) >= FIRST_CHUNK_FRAMES:
            codes = self.torch.stack(self._frames[:FIRST_CHUNK_FRAMES], dim=0)
            wavs, _ = self.model.model.speech_tokenizer.decode([{"audio_codes": codes}])
            self.torch.cuda.synchronize()
            self._t_first = time.perf_counter()
            self._first_audio = wavs[0]

    def synth(self, s: Sentence) -> Measure:
        self._frames, self._t_first, self._first_audio = [], None, None
        with VramPeak() as vp:
            t0 = time.perf_counter()
            wavs, sr = self.model.generate_custom_voice(
                text=s.text, language=QWEN_LANG[s.lang], speaker=QWEN_SPEAKER,
            )
            self.torch.cuda.synchronize()
            t1 = time.perf_counter()
        t_first = self._t_first if self._t_first is not None else t1  # 4 フレーム未満で終わった場合
        return Measure((t_first - t0) * 1000, (t1 - t0) * 1000, wavs[0], sr, vp.peak)


def measure(engine: Any, sentences: list[Sentence], runs: int) -> list[Row]:
    engine.synth(sentences[0])  # ウォームアップ（捨てる）
    rows: list[Row] = []
    wav_dir = OUT_DIR / engine.name
    wav_dir.mkdir(parents=True, exist_ok=True)
    for s in sentences:
        ms = [engine.synth(s) for _ in range(runs)]
        last = ms[-1]
        sf.write(wav_dir / f"{s.lang}_{s.n}.wav", last.audio, last.sr)
        audio_sec = len(last.audio) / last.sr
        t_total = statistics.median(m.t_total_ms for m in ms)
        peaks = [m.vram_peak_mb for m in ms if m.vram_peak_mb is not None]
        row = Row(
            engine=engine.name, lang=s.lang, n=s.n, chars=len(s.text),
            t_first_audio_ms=statistics.median(m.t_first_ms for m in ms),
            t_total_ms=t_total, audio_sec=audio_sec, rtf=t_total / 1000 / audio_sec,
            vram_peak_mb=max(peaks) if peaks else None,
            runs=[(m.t_first_ms, m.t_total_ms) for m in ms],
        )
        logger.info(
            "%s %s_%d first=%.0fms total=%.0fms audio=%.2fs rtf=%.2f vram=%s runs=%s",
            row.engine, row.lang, row.n, row.t_first_audio_ms, row.t_total_ms, row.audio_sec,
            row.rtf, row.vram_peak_mb, [(round(a), round(b)) for a, b in row.runs],
        )
        rows.append(row)
    return rows


def write_csv(rows: list[Row]) -> None:
    cols = ["engine", "lang", "n", "chars", "t_first_audio_ms", "t_total_ms", "audio_sec", "rtf", "vram_peak_mb"]
    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.engine, r.lang, r.n, r.chars, f"{r.t_first_audio_ms:.0f}", f"{r.t_total_ms:.0f}",
                f"{r.audio_sec:.2f}", f"{r.rtf:.2f}",
                "" if r.vram_peak_mb is None else f"{r.vram_peak_mb:.0f}",
            ])


def write_index(sentences: list[Sentence], rows: list[Row], engines: list[str]) -> None:
    by = {(r.engine, r.lang, r.n): r for r in rows}
    head = "".join(f"<th>{html.escape(e)}</th>" for e in engines)
    body = []
    for s in sentences:
        cells = []
        for e in engines:
            r = by.get((e, s.lang, s.n))
            if r is None:
                cells.append('<td class="na">対象外</td>')
            else:
                cells.append(
                    f'<td><audio controls preload="none" src="{e}/{s.lang}_{s.n}.wav"></audio>'
                    f'<div class="m">初回 {r.t_first_audio_ms:.0f}ms / 全体 {r.t_total_ms:.0f}ms / '
                    f'{r.audio_sec:.1f}s / RTF {r.rtf:.2f}</div></td>'
                )
        body.append(
            f"<tr><td class=\"t\"><b>{s.lang}-{s.n}</b> {html.escape(s.text)}</td>{''.join(cells)}</tr>"
        )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS 聞き比べ</title>
<style>
:root {{ --bg:#fff; --fg:#222; --mut:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1b1b1b; --fg:#eee; --mut:#aaa; --line:#444; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:sans-serif; margin:16px; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ border:1px solid var(--line); padding:8px; vertical-align:top; }}
td.t {{ max-width:28em; }} .m {{ color:var(--mut); font-size:12px; margin-top:4px; }}
td.na {{ color:var(--mut); text-align:center; }} audio {{ width:260px; max-width:100%; }}
</style></head><body>
<h1>WP00.5 TTS 聞き比べ</h1>
<p>VOICEVOX: speaker_id={VOICEVOX_SPEAKER}（アプリと同設定） / Qwen3-TTS: {QWEN_MODEL}、話者 {QWEN_SPEAKER}。
計測値は 3 回の中央値。Qwen の「初回」は先頭 {FIRST_CHUNK_FRAMES} フレームのデコード完了時刻（疑似ストリーミング）。
VOICEVOX は日本語のみ。0.6B に VoiceDesign モデルが無いため design 列は無し。</p>
<table><tr><th>文</th>{head}</tr>
{''.join(body)}
</table></body></html>
"""
    (OUT_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()
    ja = [s for s in sentences if s.lang == "ja"]

    # 指示書どおり 1 秒間隔の VRAM ログも全期間で取る
    vram_log = (OUT_DIR / "vram_log.csv").open("w")
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used", "--format=csv", "-l", "1"], stdout=vram_log,
    )
    try:
        rows = measure(VoicevoxEngine(), ja, args.runs)
        before = gpu_used_mb()
        qwen = QwenEngine()
        after = gpu_used_mb()
        logger.info("VRAM: Qwen ロード前 %.0fMB / ロード後 %.0fMB（+%.0fMB）", before, after, after - before)
        rows += measure(qwen, sentences, args.runs)
        logger.info("torch max_memory_reserved=%.0fMB", qwen.torch.cuda.max_memory_reserved() / 2**20)
    finally:
        smi.terminate()
        vram_log.close()

    write_csv(rows)
    write_index(sentences, rows, ["voicevox", "qwen3tts-0.6b"])
    logger.info("書き出し: %s, %s", OUT_DIR / "results.csv", OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
