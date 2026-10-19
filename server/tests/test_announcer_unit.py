"""announcer.py のユニットテスト。

テンプレート展開、reading の差し替え、スタイル切り替え、
TTS 読み上げ変換を検証する。
"""

from __future__ import annotations

from src.announcer import Scene, apply_tts_replacements, generate, generate_for_tts


class TestGenerate:
    """generate() のテスト。"""

    def test_opening_contains_title(self) -> None:
        text = generate(Scene.OPENING, title="テスト会議")
        assert "テスト会議" in text

    def test_presenter_intro_contains_name_and_dept(self) -> None:
        text = generate(
            Scene.PRESENTER_INTRO,
            name="田中 太郎",
            dept="マーケティング部",
            topic="上期振り返り",
        )
        assert "田中 太郎" in text
        assert "マーケティング部" in text

    def test_thank_you_contains_name(self) -> None:
        text = generate(Scene.THANK_YOU, name="佐藤 花子")
        assert "佐藤 花子" in text

    def test_closing_contains_title(self) -> None:
        text = generate(Scene.CLOSING, title="タウンホールミーティング")
        assert "タウンホールミーティング" in text

    def test_formal_style(self) -> None:
        text = generate(
            Scene.PRESENTER_INTRO,
            style="formal",
            name="佐藤 花子",
            dept="開発部",
            topic="ロードマップ",
        )
        assert "佐藤 花子" in text
        # formal スタイルでは「様」が使われる
        assert "様" in text

    def test_friendly_style_no_sama(self) -> None:
        text = generate(
            Scene.PRESENTER_INTRO,
            style="friendly",
            name="田中 太郎",
            dept="マーケティング部",
            topic="テスト",
        )
        assert "田中 太郎" in text
        # friendly スタイルでは「さん」が使われる
        assert "さん" in text

    def test_all_scenes_have_templates(self) -> None:
        """全シーン × 全スタイルでテンプレートが存在すること。"""
        for style in ("friendly", "formal"):
            for scene in Scene:
                # 最低限のキーワード引数を渡してエラーにならないことを確認
                kwargs = {
                    "name": "テスト太郎",
                    "dept": "テスト部",
                    "topic": "テスト発表",
                    "title": "テスト会議",
                }
                text = generate(scene, style=style, **kwargs)
                assert len(text) > 0


class TestGenerateForTts:
    """generate_for_tts() のテスト。"""

    def test_uses_reading_instead_of_name(self) -> None:
        text = generate_for_tts(
            Scene.THANK_YOU,
            name="田中 太郎",
            reading="たなか たろう",
        )
        # reading のスペースは除去され「たなかたろう」になる
        assert "たなかたろう" in text
        assert "田中 太郎" not in text

    def test_falls_back_to_name_without_reading(self) -> None:
        text = generate_for_tts(
            Scene.THANK_YOU,
            name="田中 太郎",
        )
        # reading 未指定時は name がそのまま使われる
        # CJK 文字間スペースは除去される
        assert "田中太郎" in text

    def test_presenter_intro_tts_uses_reading(self) -> None:
        text = generate_for_tts(
            Scene.PRESENTER_INTRO,
            name="佐藤 花子",
            reading="さとう はなこ",
            dept="開発部",
            topic="新プロダクト",
        )
        # reading のスペースは除去される
        assert "さとうはなこ" in text
        assert "佐藤 花子" not in text
        assert "開発部" in text


class TestTtsReplacements:
    """TTS テキスト変換のテスト。"""

    def test_platform_service_to_katakana(self) -> None:
        text = apply_tts_replacements("Platform Serviceの発表")
        assert "プラットフォームサービスの発表" in text
        assert "Platform Service" not in text

    def test_who_am_i_to_katakana(self) -> None:
        text = apply_tts_replacements("「Who am I?」クイズ")
        assert "フーアムアイクイズ" in text
        assert "Who am I?" not in text

    def test_cjk_space_removal(self) -> None:
        text = apply_tts_replacements("プラットフォームサービス 発表")
        assert "プラットフォームサービス発表" in text

    def test_reading_space_stripped_in_generate_for_tts(self) -> None:
        """姓名間のスペースが除去され敬称と連結されること。"""
        text = generate_for_tts(
            Scene.THANK_YOU,
            name="鈴木 一郎",
            reading="すずき いちろう",
        )
        # 「すずきいちろうさん」が一続きになる
        assert "すずきいちろうさん" in text

    def test_part_closing_replaces_platform_service(self) -> None:
        text = generate_for_tts(
            Scene.PART_CLOSING,
            title="Platform Service 発表",
        )
        assert "プラットフォームサービス" in text
        assert "Platform Service" not in text


class TestDeptReadingInTts:
    """generate_for_tts() での dept_reading 差し替えを検証する。"""

    def test_dept_reading_replaces_dept(self) -> None:
        """dept_reading 指定時に TTS テキストで dept が置換される。"""
        text = generate_for_tts(
            Scene.PRESENTER_INTRO,
            name="斉藤 英隆",
            reading="さいとう ひでたか",
            dept="Service Enablement Platform",
            dept_reading="サービスイネーブルメントプラットフォーム",
            topic="テスト",
        )
        assert "サービスイネーブルメントプラットフォーム" in text
        assert "Service Enablement Platform" not in text

    def test_no_dept_reading_keeps_original_dept(self) -> None:
        """dept_reading 未指定時は元の dept がそのまま使われる。"""
        text = generate_for_tts(
            Scene.PRESENTER_INTRO,
            name="鈴木 一郎",
            reading="すずき いちろう",
            dept="開発部",
            topic="テスト",
        )
        assert "開発部" in text
