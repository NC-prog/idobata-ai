"""asr_client.py の単体テスト。

ストリーミングテキスト抽出、言語検出・言語コード変換、拠点語彙の読み込みの検証。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.asr_client import (
    DEFAULT_LANGUAGE,
    _clean_streaming_text,
    _detect_language,
    _language_code,
    load_site_vocabulary,
)
from src.config import SERVER_ROOT


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
