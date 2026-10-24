"""英字の安全装置（WP13）: 大文字の略語は 1 応答 3 語まで、allowed_terms.yaml の語は数えず辞書に、英文は捨てる。"""

from __future__ import annotations

from src.config import SERVER_ROOT, DialogueConfig
from src.dialogue import Dialogue, foreign_tokens
from src.site_context import load_allowed_terms, load_facility

TERMS = tuple(w.surface for w in load_allowed_terms(SERVER_ROOT / "context" / "allowed_terms.yaml"))


def _bad(text: str, used: set[str] | None = None) -> list[str]:
    return foreign_tokens(text, used if used is not None else set(), allowed_terms=TERMS, allow_ascii={"aed", "jr"},
                          abbrev_max=3, run_min=3, english_words_min=4)


def test_abbreviations_up_to_three_per_reply() -> None:
    used: set[str] = set()
    assert _bad("申込書はPDFでご覧いただけます。", used) == []
    assert _bad("画面のQRコードから読み取れます。", used) == []
    assert _bad("市の公式LINEでも手続きできます。", used) == []
    assert _bad("ATMは1階にあります。", used) == ["ATM"]  # 4 語目から
    assert _bad("ＰＤＦとＱＲ。") == []  # 全角も同じ
    assert _bad("PDFは同じ語なので何度でも", {"PDF", "QR", "LINE"}) == []


def test_lowercase_and_mixed_words_are_still_banned() -> None:
    assert _bad("ご案内します、please。") == ["please"]
    assert _bad("Thank you、ありがとうございます。") == ["Thank", "you"]
    assert _bad("PLEASEと書いてあります。") == ["PLEASE"]  # 6 字の大文字は略語ではない


def test_allowed_terms_are_not_counted() -> None:
    assert _bad("館内でWi-Fiが使えます。Webでも申し込めます。") == []
    assert _bad("5kmほど先、気温は20℃です。荷物は10kgまで、長さは30cmまで。") == []
    assert _bad("JRとAEDは拠点の語なので数えない", {"PDF", "QR", "LINE"}) == []


def test_english_sentences_are_dropped() -> None:
    assert _bad("PDF QR LINE ATMのどれかで") == ["PDF", "QR", "LINE", "ATM"]  # 4 語以上
    assert _bad("詳しくは Thank you very much.") != []
    assert _bad("Wi-Fi is free.") != []  # ピリオドで終わる英字の並び


def test_dialogue_counts_per_reply_and_fixed_text_reason() -> None:
    d = Dialogue(None, DialogueConfig(), facility=load_facility("etajima", SERVER_ROOT / "context"))  # type: ignore[arg-type]
    d.begin_turn()
    assert not d._has_foreign_word("PDFとQRとATMで。")  # 江田島は LINE を拠点の語（allow_ascii）にしているので数えない
    assert d._has_foreign_word("ETCも。")  # 同じ応答の 4 語目
    d.begin_turn()  # 次の応答では数え直す
    assert not d._has_foreign_word("ETCも。")
    reasons = d.fixed_text_problems("申込書はPDFで、QRコードから。ATMとSNSでも。", [])
    assert "英字: SNS（大文字の略語は 3 語まで、ほかは allowed_terms.yaml）" in reasons


def test_allowed_terms_go_to_voicevox_dictionary_only() -> None:
    words = {w.surface: w for w in load_allowed_terms(SERVER_ROOT / "context" / "allowed_terms.yaml")}
    assert words["Wi-Fi"].reading == "ワイファイ" and not words["Wi-Fi"].asr
    assert {"Web", "km", "kg", "cm", "℃"} <= set(words)
