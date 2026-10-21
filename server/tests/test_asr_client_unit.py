"""asr_client.py の単体テスト。

ストリーミングテキスト抽出、言語検出・言語コード変換、拠点語彙の読み込みの検証。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.asr_client import (
    DEFAULT_LANGUAGE,
    ASRClient,
    _clean_streaming_text,
    _detect_language,
    _language_code,
    language_name,
    load_site_vocabulary,
    ngram_overlap,
)
from src.config import SERVER_ROOT, ASRConfig


class TestCleanStreamingText:
    """_clean_streaming_text のテスト。"""

    def test_metadata_only_language(self) -> None:
        """'language' のみの場合は空文字列を返す。"""
        assert _clean_streaming_text("language") == ""

    def test_metadata_language_name(self) -> None:
        """'language Japanese' のみの場合は空文字列を返す。"""
        assert _clean_streaming_text("language Japanese") == ""

    def test_metadata_language_chinese(self) -> None:
        """'language Chinese' のみの場合も空文字列を返す。"""
        assert _clean_streaming_text("language Chinese") == ""

    def test_empty_asr_text_tag(self) -> None:
        """<asr_text> タグが出現したが中身がまだない場合は空文字列を返す。"""
        assert _clean_streaming_text("language Japanese<asr_text>") == ""

    def test_partial_text_inside_tag(self) -> None:
        """<asr_text> タグ内のテキストを返す（閉じタグなし）。"""
        raw = "language Japanese<asr_text>こんにちは"
        assert _clean_streaming_text(raw) == "こんにちは"

    def test_complete_text_with_closing_tag(self) -> None:
        """閉じタグ付きの完全なテキストを返す。"""
        raw = "language Japanese<asr_text>こんにちは。</asr_text>"
        assert _clean_streaming_text(raw) == "こんにちは。"

    def test_multiline_text(self) -> None:
        """改行を含むテキストも正しく抽出する。"""
        raw = "language Japanese<asr_text>一行目\n二行目</asr_text>"
        assert _clean_streaming_text(raw) == "一行目\n二行目"

    def test_empty_string(self) -> None:
        """空文字列の入力には空文字列を返す。"""
        assert _clean_streaming_text("") == ""

    def test_random_text_without_tag(self) -> None:
        """<asr_text> タグなしの任意テキストには空文字列を返す。"""
        assert _clean_streaming_text("some random text") == ""

    def test_whitespace_inside_tag(self) -> None:
        """タグ内が空白のみの場合は空文字列を返す。"""
        assert _clean_streaming_text("language Japanese<asr_text>  </asr_text>") == ""


class TestDetectLanguage:
    """_detect_language のテスト。"""

    def test_japanese(self) -> None:
        assert _detect_language("language Japanese<asr_text>テスト</asr_text>") == "Japanese"

    def test_chinese(self) -> None:
        assert _detect_language("language Chinese<asr_text>嗯。</asr_text>") == "Chinese"

    def test_english(self) -> None:
        assert _detect_language("language English<asr_text>hello</asr_text>") == "English"

    def test_partial_header(self) -> None:
        """言語名のみ（<asr_text> 前でも検出可能）。"""
        assert _detect_language("language Japanese") == "Japanese"

    def test_keyword_only(self) -> None:
        """'language' キーワードのみでは None。"""
        assert _detect_language("language") is None

    def test_no_language_header(self) -> None:
        """言語ヘッダーがない場合は None。"""
        assert _detect_language("random text") is None

    def test_empty(self) -> None:
        assert _detect_language("") is None


class TestLanguageCode:
    """_language_code のテスト（検出言語は破棄せずコードで返す）。"""

    def test_japanese(self) -> None:
        assert _language_code("Japanese") == "ja"

    def test_english(self) -> None:
        assert _language_code("English") == "en"

    def test_chinese(self) -> None:
        assert _language_code("Chinese") == "zh"

    def test_vietnamese(self) -> None:
        assert _language_code("Vietnamese") == "vi"

    def test_case_insensitive(self) -> None:
        assert _language_code("JAPANESE") == "ja"

    def test_unknown_name_passthrough(self) -> None:
        """未登録の言語名は小文字化して返す。"""
        assert _language_code("Klingon") == "klingon"

    def test_none_falls_back_to_default(self) -> None:
        assert _language_code(None) == DEFAULT_LANGUAGE
        assert _language_code("None") == DEFAULT_LANGUAGE


class TestLoadSiteVocabulary:
    """拠点 YAML からの語彙読み込み。"""

    @pytest.mark.parametrize("site", ["etajima", "akiota"])
    def test_bundled_sites(self, site: str) -> None:
        words = load_site_vocabulary(site, SERVER_ROOT / "context")
        assert len(words) >= 20
        assert len(words) == len(set(words))

    def test_missing_site_returns_empty(self, tmp_path: Path) -> None:
        assert load_site_vocabulary("nowhere", tmp_path) == []

    def test_flattens_categories(self, tmp_path: Path) -> None:
        (tmp_path / "x.yaml").write_text(
            "site_name: テスト市\nfacilities: [市役所, 公民館]\ndepartments: [市民課, 市役所]\n",
            encoding="utf-8",
        )
        assert load_site_vocabulary("x", tmp_path) == ["テスト市", "市役所", "公民館", "市民課"]


class TestCleanStreamingTextProgressiveAccumulation:
    """ストリーミングのトークンが逐次蓄積される過程をシミュレートする。"""

    @pytest.fixture()
    def tokens(self) -> list[str]:
        """TASK-001 実測のストリーミングトークン列。"""
        return [
            "language",
            " Japanese",
            "<asr_text>",
            "こんにちは",
            "。",
            "本日",
            "は",
            "タウンホール",
            "です",
            "。",
            "</asr_text>",
        ]

    def test_no_leakage_before_asr_text(self, tokens: list[str]) -> None:
        """<asr_text> 出現前のトークンでは空文字列のみを返す。"""
        collected = ""
        for token in tokens[:2]:  # "language", " Japanese"
            collected += token
            assert _clean_streaming_text(collected) == ""

    def test_empty_right_after_asr_text_tag(self, tokens: list[str]) -> None:
        """<asr_text> タグ直後はまだテキストがないので空文字列。"""
        collected = "".join(tokens[:3])  # "language Japanese<asr_text>"
        assert _clean_streaming_text(collected) == ""

    def test_text_appears_after_content(self, tokens: list[str]) -> None:
        """テキストトークンが入った時点で内容が返る。"""
        collected = "".join(tokens[:4])  # ...<asr_text>こんにちは
        assert _clean_streaming_text(collected) == "こんにちは"

    def test_progressive_accumulation(self, tokens: list[str]) -> None:
        """全トークンを順次蓄積し、最終結果が正しいことを確認。"""
        collected = ""
        results: list[str] = []
        for token in tokens:
            collected += token
            result = _clean_streaming_text(collected)
            results.append(result)

        # 最初の2トークンは空
        assert results[0] == ""
        assert results[1] == ""
        # <asr_text> 直後も空
        assert results[2] == ""
        # テキストが出始めたら内容が返る
        assert results[3] == "こんにちは"
        assert results[4] == "こんにちは。"
        # 最終結果
        assert results[-1] == "こんにちは。本日はタウンホールです。"


# ------------------------------------------------------------------
# ローリングコンテキスト無効化と、コンテキスト吐き出しの破棄（WP01 追加作業）
# ------------------------------------------------------------------


def _sse_client(outputs: list[str], header: bool = True) -> tuple[ASRClient, list[dict[str, Any]]]:
    """呼ばれるたびに outputs を 1 つずつ ASR の出力として返す ASRClient。

    header=False は言語を固定したとき（言語ヘッダーはプロンプト側にあり、出力に出ない）。
    """
    import httpx

    payloads: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(req.content))
        text = outputs[len(payloads) - 1]
        body = "".join(
            f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n"
            for c in (["language Japanese", "<asr_text>"] if header else []) + [text, "</asr_text>"]
        ) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = ASRClient(ASRConfig())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, payloads


async def _final(client: ASRClient, force_language: str | None = None) -> str | None:
    audio = [np.zeros(1600, dtype=np.float32)]
    finals = [c.text async for c in client.transcribe_segment(audio, 0.0, force_language=force_language)
              if c.is_final]
    return finals[0] if finals else None


class TestContextEcho:
    def test_ngram_overlap(self) -> None:
        assert ngram_overlap("住民票の写しを取りたい", "住民票の写しを取りたいです。") == 1.0
        assert ngram_overlap("今日は寒いですね", "住民票の写しを取りたい") == 0.0
        assert ngram_overlap("あ", "あ") == 0.0  # n 文字未満

    def test_rolling_context_disabled_by_default(self) -> None:
        cfg = ASRConfig()
        assert cfg.rolling_context_enabled is False
        client = ASRClient(cfg)
        client.set_static_context(["能美", "市民課"])
        client.push_rolling_context("住民票の写しを取りたいんですけど")
        prompt = client._build_system_prompt()
        assert prompt is not None
        assert "能美" in prompt  # 静的語彙は残す
        assert "Recent transcript" not in prompt
        assert "住民票" not in prompt

    async def test_echo_of_recent_transcripts_is_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        client, _ = _sse_client([
            "住民票の写しを取りたいんですけど。",
            "最近急に寒くなりましたね。",
            # 過去の 2 発話がつながった偽の聞き取り
            "住民票の写しを取りたいんですけど、最近急に寒くなりましたね。",
            "国民健康保険の更新はどこですか。",
        ])
        assert await _final(client) == "住民票の写しを取りたいんですけど。"
        assert await _final(client) == "最近急に寒くなりましたね。"
        with caplog.at_level(logging.WARNING, logger="src.asr_client"):
            assert await _final(client) is None
        assert any("破棄" in r.message for r in caplog.records)
        assert await _final(client) == "国民健康保険の更新はどこですか。"
        await client.close()

    async def test_short_repeats_are_kept(self) -> None:
        client, _ = _sse_client(["はい、そうです。", "はい、そうです。"])
        assert await _final(client) == "はい、そうです。"
        assert await _final(client) == "はい、そうです。"
        await client.close()

    async def test_clear_forgets_recent_transcripts(self) -> None:
        text = "住民票の写しを取りたいんですけど。"
        client, _ = _sse_client([text, text])
        assert await _final(client) == text
        client.clear_rolling_context()  # 会話の区切り（次の来訪者は同じことを言ってよい）
        assert await _final(client) == text
        await client.close()


class TestForcedLanguage:
    async def test_force_language_sends_kwargs_without_vocabulary(self) -> None:
        client, payloads = _sse_client(["うん。"], header=False)
        client.set_static_context(["能美", "市民課"])
        assert await _final(client, force_language="Japanese") == "うん。"
        assert payloads[0]["chat_template_kwargs"] == {"force_language": "Japanese"}
        assert all(m["role"] != "system" for m in payloads[0]["messages"])  # 返事待ちは語彙を添えない
        await client.close()

    async def test_vocabulary_dump_is_dropped(self) -> None:
        vocab = ["江田島市", "江田島", "能美", "大柿", "沖美", "小用", "市民課"]
        client, _ = _sse_client(["江田島市, 江田島, 能美, 大柿, 沖美, 小用, 市民課.", "能美の市民課はどこですか。"])
        client.set_static_context(vocab)
        assert await _final(client) is None
        assert await _final(client) == "能美の市民課はどこですか。"
        await client.close()

    def test_language_name(self) -> None:
        assert language_name("ja") == "Japanese"
        assert language_name("xx") is None
