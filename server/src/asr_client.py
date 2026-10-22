"""Qwen3-ASR ストリーミングクライアント。

VAD で区切られた音声セグメントを base64 エンコードし、
vLLM の /v1/chat/completions (stream=true) に送信する。
SSE の delta.content を partial 字幕として yield し、
ストリーム完了時に final 字幕を yield する。

townhall では検出言語が ``ja`` 以外のセグメントを破棄していたが、井戸端AIは
多言語対応のため破棄せず、検出言語を ``TranscriptChunk.language`` で返す。
静的コンテキスト（コンテキストバイアシング）は拠点ごとの地名・施設名・課名リスト
（``server/context/<site>.yaml``）から組み立てる。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import wave
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .config import ASRConfig, config
from .schemas import TranscriptChunk
from .site_context import load_site_words

logger = logging.getLogger(__name__)

# Qwen3-ASR ストリーミング出力に含まれるタグパターン
_ASR_TEXT_PATTERN = re.compile(r"<asr_text>(.*?)(?:</asr_text>|$)", re.DOTALL)

# 言語ヘッダーパターン: "language Japanese<asr_text>..."
_LANG_DETECT_PATTERN = re.compile(r"language\s+(\w+)")

# Qwen3-ASR が出力する言語名（小文字） → ISO 639-1 コード
_LANG_NAME_TO_CODE: dict[str, str] = {
    "japanese": "ja",
    "english": "en",
    "chinese": "zh",
    "cantonese": "yue",
    "korean": "ko",
    "vietnamese": "vi",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "indonesian": "id",
    "thai": "th",
    "filipino": "fil",
    "russian": "ru",
}

# 言語ヘッダーが取れなかったときの既定言語
DEFAULT_LANGUAGE = "ja"


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """float32 の numpy 配列を WAV バイト列に変換する。"""
    buf = io.BytesIO()
    pcm16 = (samples * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _clean_streaming_text(raw: str) -> str:
    """ストリーミング出力からテキストを抽出する。

    Qwen3-ASR のストリーミング出力は ``language ...<asr_text>テキスト</asr_text>``
    形式。<asr_text> タグ内のテキストのみを返す。
    タグが未出現の場合（言語ヘッダー部分のみ）は空文字列を返す。
    """
    match = _ASR_TEXT_PATTERN.search(raw)
    if match:
        return match.group(1).strip()
    # タグが見つからない場合は先頭の言語ヘッダー部分を除去して返す
    # 典型的パターン: "ja\n<asr_text>テキスト" (タグが閉じていない場合)
    if "<asr_text>" in raw:
        idx = raw.index("<asr_text>") + len("<asr_text>")
        return raw[idx:].strip()
    # <asr_text> 未出現 → メタデータ部分（"language", "language Japanese" 等）
    return ""


def _detect_language(raw: str) -> str | None:
    """ストリーミング出力から検出言語を抽出する。

    Returns:
        検出された言語名（例: "Japanese"）。未検出の場合 None。
    """
    match = _LANG_DETECT_PATTERN.search(raw)
    if match:
        return match.group(1)
    return None


def language_name(code: str) -> str | None:
    """ISO 639-1 コードを ASR の言語名（force_language に渡す形）に変換する。未知なら None。"""
    for name, c in _LANG_NAME_TO_CODE.items():
        if c == code:
            return name.capitalize()
    return None


def _language_code(detected: str | None) -> str:
    """ASR の言語名を ISO 639-1 コードに変換する。

    未知の言語名は小文字化してそのまま返す。未検出（None）は既定言語。
    """
    # 無音・雑音では "language None" が返ることがある
    if not detected or detected.lower() == "none":
        return DEFAULT_LANGUAGE
    name = detected.lower()
    return _LANG_NAME_TO_CODE.get(name, name)


# 一致判定で無視する文字（句読点・記号・空白）
_ECHO_IGNORE_RE = re.compile(r"[\s、。，．,.！？!?「」『』（）()・…ー〜~\-]")


def _normalize_for_echo(text: str) -> str:
    return _ECHO_IGNORE_RE.sub("", text)


def ngram_overlap(text: str, reference: str, n: int = 2) -> float:
    """text の文字 n-gram のうち、reference にも現れるものの割合（0〜1）。

    句読点・空白は除いて比べる。text が n 文字未満なら 0。
    """
    a = _normalize_for_echo(text)
    b = _normalize_for_echo(reference)
    grams = [a[i : i + n] for i in range(len(a) - n + 1)]
    if not grams:
        return 0.0
    ref = {b[i : i + n] for i in range(len(b) - n + 1)}
    return sum(1 for g in grams if g in ref) / len(grams)


def load_site_vocabulary(site: str, context_dir: Path, avatar_name: str = "") -> list[str]:
    """拠点の YAML（``<context_dir>/<site>.yaml``）から語彙（表記）のリストを読む。

    YAML の書式は ``site_context.load_site_words`` を参照（読み付きの語は表記だけを使う）。
    ファイルが無い・壊れている場合は空リストを返す（ASR は語彙なしで動く）。
    アバターの呼び名を渡すと先頭に置く（「いどちゃん」が「伊豆ちゃん」になったため）。
    """
    return [w.surface for w in load_site_words(site, context_dir, avatar_name) if w.asr]


class ASRClient:
    """Qwen3-ASR ストリーミングクライアント。"""

    def __init__(self, cfg: ASRConfig | None = None) -> None:
        self._cfg = cfg or config.asr
        self._client = httpx.AsyncClient(timeout=self._cfg.timeout)

        # 静的コンテキスト（拠点の語彙、set_static_context で設定）
        self._static_context: str = ""
        self._vocabulary: list[str] = []

        # ローリングコンテキスト（直前の確定転写バッファ）
        self._rolling_segments: list[str] = []
        # 直前の確定転写（コンテキスト吐き出しの判定用。ローリングコンテキストの有効・無効に依らない）
        self._recent_finals: deque[str] = deque(maxlen=self._cfg.echo_check_segments)

    def set_static_context(self, vocabulary: list[str]) -> None:
        """拠点の地名・施設名・課名から静的コンテキストを設定する。

        Args:
            vocabulary: ASR 補正用の固有名詞リスト
        """
        if not self._cfg.context_biasing or not vocabulary:
            self._static_context = ""
            self._vocabulary = []
            return

        self._vocabulary = list(vocabulary)
        self._static_context = "Proper nouns: " + ", ".join(vocabulary)
        logger.info("ASR 静的コンテキスト設定: %d 語, %d 文字",
                    len(vocabulary), len(self._static_context))

    def push_rolling_context(self, text: str) -> None:
        """確定転写テキストをローリングバッファに追加する。"""
        if not self._cfg.rolling_context_enabled or not text:
            return
        self._rolling_segments.append(text)
        # セグメント数上限
        max_segs = self._cfg.rolling_context_segments
        if len(self._rolling_segments) > max_segs:
            self._rolling_segments = self._rolling_segments[-max_segs:]

    def clear_rolling_context(self) -> None:
        """ローリングバッファと直前の確定転写をクリアする（会話の切れ目・リセット時）。"""
        self._rolling_segments.clear()
        self._recent_finals.clear()
        logger.debug("ASR ローリングコンテキストをクリア")

    def _build_system_prompt(self) -> str | None:
        """静的コンテキスト + ローリングコンテキストからシステムプロンプトを組み立てる。"""
        parts: list[str] = []

        if self._static_context:
            parts.append(self._static_context)

        if self._cfg.rolling_context_enabled and self._rolling_segments:
            rolling = "".join(self._rolling_segments)
            # 文字数上限でトリム（末尾を残す）
            max_chars = self._cfg.rolling_context_max_chars
            if len(rolling) > max_chars:
                rolling = rolling[-max_chars:]
            parts.append("Recent transcript: " + rolling)

        if not parts:
            return None
        return "\n".join(parts)

    async def transcribe_segment(
        self,
        audio_frames: list[np.ndarray],
        t_speech_start: float,
        sample_rate: int = 16000,
        force_language: str | None = None,
    ) -> AsyncIterator[TranscriptChunk]:
        """音声セグメントをストリーミング文字起こしする。

        Args:
            audio_frames: VAD で蓄積された音声フレームのリスト
            t_speech_start: 発話開始時の monotonic 時刻
            sample_rate: サンプリングレート
            force_language: 言語を固定する場合の言語名（例: "Japanese"）。返事待ちの短い発話用。
                ASR サーバのテンプレート（deploy/standalone/asr/chat_template.jinja）が要る

        Yields:
            TranscriptChunk (partial/final)
        """
        # フレーム結合 → WAV バイト列
        audio = np.concatenate(audio_frames)
        wav_bytes = _make_wav_bytes(audio, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_uri = f"data:audio/wav;base64,{audio_b64}"

        messages: list[dict[str, Any]] = []

        # システムプロンプト（コンテキストバイアシング）
        # 言語を固定する返事待ち（うん・はい）は語彙が要らない。短い音声に語彙を添えると、
        # 語彙の一覧をそのまま吐き出すことがある
        system_prompt = None if force_language else self._build_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": data_uri},
                }
            ],
        })

        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "stream": True,
            "messages": messages,
            "temperature": self._cfg.temperature,
        }
        if force_language:
            payload["chat_template_kwargs"] = {"force_language": force_language}

        # 言語を固定したときは、言語ヘッダーがプロンプト側にあるので出力に現れない
        collected = f"language {force_language}<asr_text>" if force_language else ""
        language = DEFAULT_LANGUAGE
        language_checked = False

        try:
            async with self._client.stream(
                "POST",
                f"{self._cfg.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": "Bearer dummy",
                    "Content-Type": "application/json",
                },
            ) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        continue

                    collected += content

                    # 言語検出: <asr_text> 出現時に言語ヘッダーを読む（破棄はしない）
                    if not language_checked and "<asr_text>" in collected:
                        language_checked = True
                        language = _language_code(_detect_language(collected))
                        logger.debug("検出言語: %s", language)

                    cleaned = _clean_streaming_text(collected)

                    if cleaned:
                        yield TranscriptChunk(
                            text=cleaned,
                            is_final=False,
                            t_speech_start=t_speech_start,
                            t_emitted=time.monotonic(),
                            language=language,
                        )

        except httpx.HTTPStatusError as exc:
            logger.error("ASR HTTP エラー (%s)。セグメントを破棄します。", exc.response.status_code)
            return
        except httpx.ConnectError:
            logger.error("ASR に接続できません (%s)。セグメントを破棄します。", self._cfg.base_url)
            return
        except Exception:
            logger.exception("ASR 呼び出し中に予期しないエラー。セグメントを破棄します。")
            return

        # ストリーム完了 → final チャンク
        final_text = _clean_streaming_text(collected)
        if final_text and self._is_vocabulary_dump(final_text):
            logger.warning("ASR 出力が語彙の一覧のため破棄しました（コンテキストの吐き出しの疑い）")
        elif final_text and self._is_context_echo(final_text):
            logger.warning(
                "ASR 出力が直前の確定転写とほぼ同じため破棄しました（コンテキストの吐き出しの疑い）"
            )
            # 本文は出さない（氏名・電話番号の答えかもしれない。WP04）
            logger.debug("破棄した ASR 出力: %d 文字", len(final_text))
        elif final_text:
            self._recent_finals.append(final_text)
            yield TranscriptChunk(
                text=final_text,
                is_final=True,
                t_speech_start=t_speech_start,
                t_emitted=time.monotonic(),
                language=language,
            )
        else:
            logger.warning("ASR が空のテキストを返しました")

    def _is_vocabulary_dump(self, text: str) -> bool:
        """静的コンテキストの語彙を並べただけの出力か（短い・曖昧な音声で起きる）。"""
        hits = sum(1 for w in self._vocabulary if w in text)
        return hits >= self._cfg.vocabulary_dump_min_words

    def _is_context_echo(self, text: str) -> bool:
        """直前の確定転写のいずれか（またはそれらをつなげたもの）の焼き直しか。"""
        cfg = self._cfg
        if len(_normalize_for_echo(text)) < cfg.echo_min_chars or not self._recent_finals:
            return False
        refs = [*self._recent_finals, "".join(self._recent_finals)]
        return any(
            ngram_overlap(text, ref, cfg.echo_ngram) >= cfg.echo_overlap_threshold for ref in refs
        )

    async def close(self) -> None:
        """HTTP クライアントを終了する。"""
        await self._client.aclose()
        logger.info("ASR クライアントを終了しました")
