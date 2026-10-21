"""dialogue.py のテスト（LLM はスタブ）。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from src.config import DialogueConfig
from src.dialogue import Dialogue, SentenceSplitter, _ThinkFilter


class StubLLM:
    """chat_stream で決まった断片を返す。fail_after 個返したあと例外を投げる。"""

    def __init__(
        self,
        deltas: list[str] | None = None,
        fail_after: int | None = None,
        rephrase: str | Exception = "日本語で言い直しました。",
    ) -> None:
        self.deltas = deltas if deltas is not None else ["そうなん", "ですね。"]
        self.fail_after = fail_after
        self.rephrase = rephrase
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        self.chat_calls: list[list[dict[str, Any]]] = []

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.chat_calls.append(messages)
        if isinstance(self.rephrase, Exception):
            raise self.rephrase
        return self.rephrase

    def chat_stream(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        self.calls.append((model, messages, kwargs))
        for i, d in enumerate(self.deltas):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("boom")
            yield d
        if self.fail_after is not None and self.fail_after >= len(self.deltas):
            raise RuntimeError("boom")


async def _collect(d: Dialogue, text: str, language: str = "ja") -> list[str]:
    return [s async for s in d.respond(text, language)]


async def test_respond_uses_system_prompt_and_disables_thinking() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig(), model="/models/llm")
    assert await _collect(d, "こんにちは") == ["そうなんですね。"]
    model, messages, kwargs = llm.calls[0]
    assert model == "/models/llm"
    assert messages[0]["role"] == "system"
    assert "案内役" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("【来訪者の発話】\nこんにちは")
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert kwargs["temperature"] == 0.5
    assert kwargs["presence_penalty"] == 0.3


async def test_yields_sentence_by_sentence_and_keeps_full_reply_in_history() -> None:
    llm = StubLLM(["こんにちは。今日は", "寒いですね？", "お体", "に気をつけて"])
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "やあ") == ["こんにちは。", "今日は寒いですね？", "お体に気をつけて"]
    await _collect(d, "次")
    _, messages, _ = llm.calls[-1]
    assert messages[2] == {
        "role": "assistant", "content": "こんにちは。今日は寒いですね？お体に気をつけて",
    }


async def test_history_keeps_only_recent_turns() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig(history_turns=6))
    for i in range(10):
        await _collect(d, f"発話{i}")
    assert d.history_len == 6
    _, messages, _ = llm.calls[-1]
    # system + 6 ターン × 2 + 今回の user
    assert len(messages) == 1 + 12 + 1
    assert messages[1]["content"] == "発話3"  # 履歴は発話だけ（参考資料を付けない）


async def test_non_japanese_gets_fixed_reply_without_llm() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "Where is the city hall?", language="en") == [
        "申し訳ありません、外国語でのご案内は準備中です。"
    ]
    assert llm.calls == []


def test_non_japanese_is_tagged_in_messages() -> None:
    d = Dialogue(StubLLM(), DialogueConfig())
    assert d.build_messages("hello", "en")[-1]["content"].endswith("[language: en] hello")


async def test_llm_failure_returns_fallback_without_history() -> None:
    d = Dialogue(StubLLM(fail_after=0), DialogueConfig())
    reply = await _collect(d, "こんにちは")
    assert len(reply) == 1
    assert "もう一度" in reply[0]
    assert d.history_len == 0


async def test_failure_midway_keeps_spoken_sentences_only() -> None:
    llm = StubLLM(["一文目です。二文", "目の途中"], fail_after=2)
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "こんにちは") == ["一文目です。"]
    await _collect(d, "次")
    assert llm.calls[-1][1][2]["content"] == "一文目です。"


async def test_empty_reply_returns_fallback() -> None:
    d = Dialogue(StubLLM(["  ", "\n"]), DialogueConfig())
    reply = await _collect(d, "こんにちは")
    assert "もう一度" in reply[0]
    assert d.history_len == 0


async def test_empty_input_is_ignored() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "   ") == []
    assert llm.calls == []


async def test_reset_clears_history() -> None:
    d = Dialogue(StubLLM(), DialogueConfig())
    await _collect(d, "a")
    d.reset()
    assert d.history_len == 0


async def test_leading_think_block_is_dropped() -> None:
    d = Dialogue(StubLLM(["<thi", "nk>\n考え中</th", "ink>\n\nはい。"]), DialogueConfig())
    assert await _collect(d, "a") == ["はい。"]


# ------------------------------------------------------------------
# SentenceSplitter
# ------------------------------------------------------------------


def _split(deltas: list[str]) -> list[str]:
    cfg = DialogueConfig()
    sp = SentenceSplitter(cfg.sentence_terminators, cfg.sentence_closers)
    out: list[str] = []
    for d in deltas:
        out += sp.feed(d)
    tail = sp.flush()
    return out + ([tail] if tail else [])


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        (["あ。い。う"], ["あ。", "い。", "う"]),
        (["本当", "ですか？！", "はい"], ["本当ですか？！", "はい"]),
        (["「どうぞ。", "」と言った。"], ["「どうぞ。」", "と言った。"]),
        (["一行目\n\n二行目"], ["一行目", "二行目"]),
        (["Hello. ", "How are", " you? Fine."], ["Hello.", "How are you?", "Fine."]),
        (["It costs 1.5 yen."], ["It costs 1.5 yen."]),
        (["。"], ["。"]),
    ],
)
def test_sentence_splitter(deltas: list[str], expected: list[str]) -> None:
    assert _split(deltas) == expected


def test_splitter_waits_for_possible_continuation() -> None:
    cfg = DialogueConfig()
    sp = SentenceSplitter(cfg.sentence_terminators, cfg.sentence_closers)
    # 末尾の「？」の後に「！」が続くかもしれないので、まだ確定しない
    assert sp.feed("ですか？") == []
    assert sp.feed("！次") == ["ですか？！"]


def test_first_sentence_is_also_cut_at_comma() -> None:
    cfg = DialogueConfig()
    sp = SentenceSplitter(
        cfg.sentence_terminators, cfg.sentence_closers,
        cfg.first_chunk_soft_terminators, cfg.first_chunk_min_chars,
    )
    out = sp.feed("ええ、住民票の写しは、二階の市民課です。手続きは、窓口で。") + [sp.flush()]
    # 1 文目だけ読点で切る（2 文字以上）。2 文目以降は読点で切らない
    assert out == ["ええ、", "住民票の写しは、二階の市民課です。", "手続きは、窓口で。"]
    sp2 = SentenceSplitter(
        cfg.sentence_terminators, cfg.sentence_closers,
        cfg.first_chunk_soft_terminators, cfg.first_chunk_min_chars,
    )
    assert sp2.feed("あ、そうですね。") + [sp2.flush()] == ["あ、そうですね。"]  # 1 文字では切らない


def test_think_filter_passes_plain_text() -> None:
    f = _ThinkFilter()
    assert f.feed("こん") == "こん"
    assert f.feed("<think>") == "<think>"  # 先頭以外のタグには触らない


# ------------------------------------------------------------------
# WP03: 英字の混じった文の言い直し・出典の印・参考資料
# ------------------------------------------------------------------


async def test_sentence_with_english_word_is_rephrased_once() -> None:
    llm = StubLLM(["窓口でcheckしてください。", "以上です。"], rephrase="窓口で確認してください。")
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "a") == ["窓口で確認してください。", "以上です。"]
    assert len(llm.chat_calls) == 1
    assert llm.chat_calls[0][1]["content"] == "窓口でcheckしてください。"
    # 履歴にも言い直した文が入る
    await _collect(d, "b")
    assert llm.calls[-1][1][2]["content"] == "窓口で確認してください。以上です。"


async def test_sentence_still_english_after_rephrase_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    llm = StubLLM(["Please wait。", "少々お待ちください。"], rephrase="Please wait.")
    d = Dialogue(llm, DialogueConfig())
    with caplog.at_level(logging.WARNING, logger="src.dialogue"):
        assert await _collect(d, "a") == ["少々お待ちください。"]
    assert any("読み上げません" in r.message for r in caplog.records)


async def test_short_or_allowed_ascii_is_kept() -> None:
    from src.site_context import Facility

    fac = Facility(name="江田島市役所", location_phrase="1階ロビー", allow_ascii=("LINE",))
    llm = StubLLM(["JRやLINEで予約できます。"])
    d = Dialogue(llm, DialogueConfig(), facility=fac)
    assert await _collect(d, "a") == ["JRやLINEで予約できます。"]
    assert llm.chat_calls == []


async def test_source_marker_is_stripped_and_cited() -> None:
    from src.knowledge import Reference

    refs = [Reference("資料1", "https://example.jp/1", "本文", 0.9, 1.0, "official", "市のホームページ"),
            Reference("資料2", "https://example.jp/2", "本文", 0.9, 1.0, "semi", "観光協会のホームページ"),
            Reference("資料3", "https://example.jp/3", "本文", 0.9, 1.0, "semi", "観光協会のホームページ")]

    class FakeKB:
        def search(self, text: str) -> Any:
            from src.knowledge import SearchResult

            return SearchResult(refs, 12.0)

    class FakeWeather:
        def lines(self, now: Any) -> list[str]:
            return ["今日（10月21日）の広島県南部は「晴れ」の予報です。"]

    llm = StubLLM(["秋祭りは土曜日です[1]。", "詳しくは市のホームページにも載っています [出典:1，３]。"])
    d = Dialogue(llm, DialogueConfig(), knowledge=FakeKB(),  # type: ignore[arg-type]
                 weather=FakeWeather())  # type: ignore[arg-type]
    out = await _collect(d, "お祭りはありますか")
    # LLM が書いた出典の一言は捨て、使った資料の出どころから足し直す
    assert out == ["秋祭りは土曜日です。", "詳しくは市のホームページや観光協会のホームページにも載っています。"]
    assert [r.title for r in d.last_cited] == ["資料1", "資料3"]
    assert d.last_search_ms == 12.0
    system = llm.calls[0][1][0]["content"]
    turn = llm.calls[0][1][-1]["content"]
    assert "[1] (出典: 資料1／市のホームページ)" in turn and "出典: 資料1" not in system
    assert "[出典:1,3]" in turn
    assert "広島県南部は「晴れ」" in system


async def test_no_kb_means_no_references() -> None:
    llm = StubLLM()
    d = Dialogue(llm, DialogueConfig())
    await _collect(d, "a")
    system = llm.calls[0][1][0]["content"]
    turn = llm.calls[0][1][-1]["content"]
    assert turn.startswith("【参考資料】\nなし")
    # 参考資料が無いターンは出典の一言を言わせない
    assert "ホームページにも載っています" not in system + turn
    assert d.last_search_ms is None


async def test_banned_phrase_sentence_is_dropped_and_leaks_rewritten() -> None:
    llm = StubLLM(["まずはお住まいの市区町村に確認してください。", "【参考資料】には載っていません。",
                   "保健医療課でご案内します。"])
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "国保") == ["手元の資料には載っていません。", "保健医療課でご案内します。"]


async def test_outside_places_are_not_spoken() -> None:
    from src.site_context import Facility

    fac = Facility(name="江田島市役所", location_phrase="1階ロビー", outside_places=("宮島",))
    llm = StubLLM(["例えば宮島がおすすめです。", "商工観光課でご案内しています。"])
    d = Dialogue(llm, DialogueConfig(), facility=fac)
    assert await _collect(d, "観光地は") == ["商工観光課でご案内しています。"]


async def test_emergency_number_is_appended_when_missing() -> None:
    from src.site_context import EmergencyLine, Facility

    fac = Facility(name="江田島市役所", location_phrase="1階ロビー", emergency_lines=(
        EmergencyLine("マイナンバー総合フリーダイヤル（24時間）", "0120-95-0178"),))
    d = Dialogue(StubLLM(["まずマイナンバー総合フリーダイヤルに電話してください。"]),
                 DialogueConfig(), facility=fac)
    assert await _collect(d, "なくした") == [
        "まずマイナンバー総合フリーダイヤルに電話してください。",
        "マイナンバー総合フリーダイヤルの番号は、0120-95-0178です。",
    ]
    d2 = Dialogue(StubLLM(["マイナンバー総合フリーダイヤル、０１２０－９５－０１７８へどうぞ。"]),
                  DialogueConfig(), facility=fac)
    out = await _collect(d2, "なくした")
    assert not any("番号は" in x for x in out)  # 全角で言っていれば足さない


async def test_chinese_simplified_kanji_is_rephrased() -> None:
    llm = StubLLM(["介護保险の相談は高齢介護課です。"], rephrase="介護保険の相談は高齢介護課です。")
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "介護") == ["介護保険の相談は高齢介護課です。"]
    assert len(llm.chat_calls) == 1
    # 日本語の漢字（﨑 などの互換漢字を含む）や記号では言い直さない
    llm2 = StubLLM(["にこ♡にこハウスは三﨑さんの近くです。"])
    assert await _collect(Dialogue(llm2, DialogueConfig()), "a") == ["にこ♡にこハウスは三﨑さんの近くです。"]
    assert llm2.chat_calls == []


async def test_unknown_facility_location_is_replaced() -> None:
    from src.site_context import BuildingFacility, Facility

    fac = Facility(name="江田島市役所", location_phrase="1階ロビー",
                   facilities_in_building=(BuildingFacility("AED", "1階ロビー"),))
    llm = StubLLM(["トイレは1階ロビーの奥にあります。", "AEDは1階ロビーにあります。",
                   "エレベーターは近くの職員にお尋ねください。"])
    d = Dialogue(llm, DialogueConfig(), facility=fac)
    assert await _collect(d, "設備") == [
        "トイレの場所は、近くの職員にお尋ねください。",   # 登録なし → 差し替え
        "AEDは1階ロビーにあります。",                   # 登録済み → そのまま
        "エレベーターは近くの職員にお尋ねください。",     # 場所を言っていない → そのまま
    ]


def test_search_query_gets_concrete_dates() -> None:
    from datetime import datetime

    from src.knowledge import JST

    d = Dialogue(StubLLM(), DialogueConfig(), now=lambda: datetime(2026, 10, 21, 10, 0, tzinfo=JST))
    assert d._expand_dates("来週末に何かイベントはありますか").endswith(
        " 10月31日 11月1日 令和8年10月 令和8年11月")
    assert d._expand_dates("今週末、何かお祭りありますか").endswith(" 10月24日 10月25日 令和8年10月")
    assert d._expand_dates("ごみの出し方") == "ごみの出し方"


async def test_question_about_unregistered_facility_skips_llm() -> None:
    from src.site_context import BuildingFacility, Facility

    fac = Facility(name="江田島市役所", location_phrase="1階ロビー",
                   facilities_in_building=(BuildingFacility("AED", "1階ロビー"),))
    llm = StubLLM(["AEDは1階ロビーにあります。"])
    d = Dialogue(llm, DialogueConfig(), facility=fac)
    assert await _collect(d, "エレベーターはありますか") == ["エレベーターの場所は、近くの職員にお尋ねください。"]
    assert await _collect(d, "トイレに行きたいんですが") == ["トイレの場所は、近くの職員にお尋ねください。"]
    assert llm.calls == []
    assert await _collect(d, "AEDはどこですか") == ["AEDは1階ロビーにあります。"]  # 登録済みは LLM に任せる
    assert len(llm.calls) == 1


def test_latest_dated() -> None:
    from datetime import date

    from src.dialogue import latest_dated

    assert latest_dated("令和8年1月10日（土）と令和８年１月２５日（日）") == date(2026, 1, 25)
    assert latest_dated("2026年10月31日に開催") == date(2026, 10, 31)
    assert latest_dated("毎週火曜日は休館") is None


async def test_past_event_refs_dropped_for_date_questions() -> None:
    from datetime import datetime

    from src.knowledge import JST, Reference, SearchResult

    past = Reference("街の防災活動", "u1", "とんど祭り 令和8年1月25日（日）", 0.9, 1.0)
    future = Reference("美術展", "u2", "令和8年10月31日から11月1日", 0.9, 1.0)
    undated = Reference("図書館", "u3", "毎週火曜日は休館", 0.9, 1.0)

    class KB:
        def search(self, q: str) -> SearchResult:
            return SearchResult([past, future, undated], 5.0)

    d = Dialogue(StubLLM(), DialogueConfig(), knowledge=KB(),
                 now=lambda: datetime(2026, 10, 21, 10, 0, tzinfo=JST))
    await _collect(d, "今週末、何かお祭りありますか")
    assert [r.title for r in d.last_references] == ["美術展", "図書館"]
    await _collect(d, "とんど祭りのことを知りたい")  # 日付を聞いていなければ外さない
    assert len(d.last_references) == 3


async def test_bracketless_source_marker_is_stripped() -> None:
    from src.knowledge import Reference, SearchResult

    ref = Reference("印鑑登録の手続き", "u", "本文", 0.9, 1.0, "official", "市のホームページ")

    class KB:
        def search(self, q: str) -> SearchResult:
            return SearchResult([ref], 5.0)

    d = Dialogue(StubLLM(["登録する印鑑が要ります。出典:1"]), DialogueConfig(), knowledge=KB())
    assert await _collect(d, "印鑑登録") == ["登録する印鑑が要ります。", "詳しくは市のホームページにも載っています。"]


async def test_leading_echo_of_visitor_is_dropped() -> None:
    llm = StubLLM(["最近、急に寒くなりましたね。", "お体に気をつけてくださいね。"])
    d = Dialogue(llm, DialogueConfig())
    out = await _collect(d, "最近、急に寒くなりましたね。朝晩は特に冷えますよ。")
    assert out == ["お体に気をつけてくださいね。"]


async def test_greeting_back_is_not_treated_as_echo() -> None:
    llm = StubLLM(["こんにちは。", "ようこそ。"])
    d = Dialogue(llm, DialogueConfig())
    assert await _collect(d, "こんにちは。ちょっと寄ってみました。") == ["こんにちは。", "ようこそ。"]
