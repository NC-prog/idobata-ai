"""お知らせ（WP11 A）の純粋な関数のテスト（ネットワーク・LLM・VOICEVOX には繋がない）。"""

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path
from typing import Any

import news_classify
import news_collect
import news_summarize
from PIL import Image

from src.config import NewsConfig, config
from src.knowledge import JST
from src.news import NewsRecord, load_records, load_rules, save_record

RULES = load_rules(config.news.rules_path)
CFG = config.news


def _record(**kw: Any) -> NewsRecord:
    base: dict[str, Any] = {"id": "r1", "title": "題名", "url": "https://www.akiota.jp/soshiki/1/1.html",
                            "published": "2026-10-20", "body": "本文" * 40, "content_hash": "h1",
                            "fetched_at": "2026-10-20T06:00:00+09:00"}
    base.update(kw)
    return NewsRecord(**base)


# ---- 分類（規則）----


def test_rule_skip_words_win_over_introduce_in_title() -> None:
    # 「入札」は「募集」より先に見る（題名に両方あれば紹介しない）
    assert news_classify.rule_relevance("指定管理者の募集（入札）", "", RULES) == ("skip", "題名に「入札」")
    assert news_classify.rule_relevance("町道の通行止めのお知らせ", "", RULES) == ("introduce", "題名に「通行止め」")


def test_rule_title_before_body() -> None:
    # 題名に紹介する語があれば、本文の紹介しない語より優先する
    got = news_classify.rule_relevance("秋祭りを開催します", "予算の範囲で", RULES)
    assert got is not None and got[0] == "introduce"
    # 題名で決まらなければ本文の先頭で
    got = news_classify.rule_relevance("お知らせ", "令和9年度予算の編成方針です", RULES)
    assert got is not None and got[0] == "skip"
    assert news_classify.rule_relevance("お知らせ", "町の取り組みです", RULES) is None


def test_classify_falls_to_llm_only_when_rules_undecided() -> None:
    class FakeLLM:
        calls = 0

        def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
            FakeLLM.calls += 1
            return "introduce\n多くの住民が使う乗り物の割り引きなので。"

    r = _record(title="町の取り組みについて", body="町の取り組みを紹介します。" * 10)
    assert news_classify.classify_record(r, RULES, FakeLLM(), CFG, date(2026, 10, 24))  # type: ignore[arg-type]
    assert (r.relevance, r.relevance_by, FakeLLM.calls) == ("introduce", "llm", 1)
    r2 = _record(title="断水のお知らせ")
    assert news_classify.classify_record(r2, RULES, FakeLLM(), CFG, date(2026, 10, 24))  # type: ignore[arg-type]
    assert (r2.relevance, r2.relevance_by, FakeLLM.calls) == ("introduce", "rule", 1)
    # LLM が要るのに無ければ分類しない（次の夜にやり直す）
    assert not news_classify.classify_record(_record(title="町の取り組み"), RULES, None, CFG, date(2026, 10, 24))


def test_closed_or_old_articles_are_skipped() -> None:
    today = date(2026, 10, 24)
    ended = _record(title="もみじまつりを開催します", period={"start": "2026-10-11", "end": "2026-10-11"})
    news_classify.classify_record(ended, RULES, None, CFG, today)
    assert ended.relevance == "skip" and "期間" in ended.relevance_reason
    closed = _record(title="料理教室", body="開催日 10月28日\n申込期限 10月16日（金）\n" + "内容" * 30)
    news_classify.classify_record(closed, RULES, None, CFG, today)
    assert closed.relevance == "skip" and "締め切り" in closed.relevance_reason
    old = _record(title="計画の案内", published="2026-05-01", body="開催" + "内容" * 40)
    news_classify.classify_record(old, RULES, None, CFG, today)
    assert old.relevance == "skip" and "日たっている" in old.relevance_reason
    # 古くても、期間の終わりがこれから先なら紹介する
    ongoing = _record(title="乗合割引を開始します", published="2026-05-01", period={"start": "2026-05-01",
                                                                             "end": "2027-03-31"})
    news_classify.classify_record(ongoing, RULES, None, CFG, today)
    assert ongoing.relevance == "introduce"
    thin = _record(title="広報の一覧（募集）", body="PDF")
    news_classify.classify_record(thin, RULES, None, CFG, today)
    assert thin.relevance == "skip"


def test_category_order() -> None:
    assert RULES.category("町道の通行止め（イベント開催のため）", "") == "通行止め"
    assert RULES.category("断水のお知らせ", "") == "ライフライン"
    assert RULES.category("お知らせ", "ごみの収集日が変わります") == "ごみ"
    assert RULES.category("町の話", "") == "その他"
    assert RULES.weight("通行止め") > RULES.weight("施設") > RULES.weight("イベント")


# ---- 一覧・期限 ----


def test_parse_rss1_and_atom() -> None:
    rss = """<rdf:RDF><item rdf:about="https://www.akiota.jp/soshiki/25/19246.html">
      <title>サイクルフェスの開催</title><link>https://www.akiota.jp/soshiki/25/19246.html</link>
      <dc:date>2026-10-23T09:00:00+09:00</dc:date><nc:event_sdate>2026-11-08</nc:event_sdate>
      <nc:event_edate></nc:event_edate></item></rdf:RDF>"""
    items = news_collect.parse_feed(rss, "https://www.akiota.jp/rss/10/list1.xml")
    assert [(i.url, i.published, i.start, i.end) for i in items] == [
        ("https://www.akiota.jp/soshiki/25/19246.html", "2026-10-23", "2026-11-08", "")]
    atom = """<feed><entry><title>断水のお知らせ</title>
      <link rel="alternate" href="/cms/articles/show/12426"/><published>2026-10-20T10:00:00+09:00</published></entry></feed>"""
    items = news_collect.parse_feed(atom, "https://www.city.etajima.hiroshima.jp/cms/atom/news")
    assert items[0].url == "https://www.city.etajima.hiroshima.jp/cms/articles/show/12426"
    assert items[0].published == "2026-10-20"


def test_period_and_expiry() -> None:
    pub = date(2026, 10, 21)
    feed = news_collect.FeedItem("u", start="2026-11-08", end="2026-11-09")
    assert news_collect.find_period(feed, "", "", pub, RULES) == {"start": "2026-11-08", "end": "2026-11-09"}
    item = news_collect.FeedItem("u")
    body = "■実施期間 令和８年１０月２０日（火）から令和９年３月７日（日）まで"
    assert news_collect.find_period(item, "", body, pub, RULES) == {"start": "2026-10-20", "end": "2027-03-07"}
    # 年の無い日付は掲載日から（10 月の記事の「1月10日」は翌年）
    assert news_collect.find_period(item, "", "日時 1月10日（日）午後1時", pub, RULES)["end"] == "2027-01-10"
    # 「10月1日から開始」は始まりだけ（終わりにしない）
    assert news_collect.find_period(item, "", "令和8年10月1日から乗合割引を開始します", date(2026, 9, 14),
                                    RULES) == {"start": "2026-10-01"}
    # 期間の語の無い行の日付・遠い過去の日付は使わない
    assert news_collect.find_period(item, "", "平成25年8月1日に制度ができました", pub, RULES) == {}
    # 期限: 期間の終わりの翌日、無ければ取得から default_ttl_days 日
    assert news_collect.expires_on({"end": "2026-11-09"}, date(2026, 10, 21), CFG) == "2026-11-10"
    assert news_collect.expires_on({"start": "2026-10-01"}, date(2026, 10, 21), CFG) == "2026-11-04"
    assert news_collect.expires_on({}, date(2026, 10, 21), NewsConfig(default_ttl_days=3)) == "2026-10-24"


def test_prune_removes_only_expired_and_old(tmp_path: Path) -> None:
    today = date(2026, 12, 1)
    save_record(tmp_path, _record(id="old", expires_at="2026-11-01", fetched_at="2026-10-20T06:00:00+09:00"))
    (tmp_path / "old.jpg").write_bytes(b"x")
    save_record(tmp_path, _record(id="young", expires_at="2026-11-30", fetched_at="2026-11-25T06:00:00+09:00"))
    save_record(tmp_path, _record(id="live", expires_at="2027-03-01", fetched_at="2026-09-01T06:00:00+09:00"))
    assert news_collect.prune(tmp_path, today, CFG) == 1
    assert sorted(r.id for r in load_records(tmp_path)) == ["live", "young"]
    assert not (tmp_path / "old.jpg").exists()


def test_clean_body_drops_pdf_boilerplate() -> None:
    body = "本文\n\n断水します。\n\nPDF形式のファイルをご覧いただく場合には、Adobe社が提供するAdobe Readerが必要です。\n" \
           "回収の案内 \\[PDFファイル／159KB\\]\n＜外部リンク＞"
    assert news_collect.clean_body(body, CFG) == "断水します。\n\n回収の案内"


# ---- 画像 ----


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(buf, "PNG")
    return buf.getvalue()


def test_image_conditions() -> None:
    page = "https://www.akiota.jp/soshiki/25/19246.html"
    allow = ["www.akiota.jp", "cs-akiota.or.jp"]
    assert news_collect.image_allowed("https://www.akiota.jp/uploaded/image/1.jpg", page, allow)
    # 許可ドメイン外
    assert not news_collect.image_allowed("https://example.com/1.jpg", page, allow)
    # 許可ドメイン内でも記事と別のサイト
    assert not news_collect.image_allowed("https://cs-akiota.or.jp/1.jpg", page, allow)
    # 小さい画像は使わない。大きい画像は長辺を縮める
    assert news_collect.fit_image(_png(300, 200), CFG) is None
    big = news_collect.fit_image(_png(2560, 1440), CFG)
    assert big is not None
    img = Image.open(io.BytesIO(big))
    assert max(img.size) == CFG.image_max_edge and img.format == "JPEG"
    assert news_collect.fit_image(b"not an image", CFG) is None


def test_image_candidates_og_then_first_body_image() -> None:
    html = """<html><head><meta property="og:image" content="/uploaded/og.jpg"></head><body>
      <img src="/img/common/pc_logo.svg"><h1>題名</h1><div id="main_body"><img src="/img/common/icon.png">
      <img src="/uploaded/image/photo.jpg"></div><footer><img src="/img/f.png"></footer></body></html>"""
    page = "https://www.akiota.jp/soshiki/1/1.html"
    assert news_collect.image_candidates(html, page) == [
        "https://www.akiota.jp/uploaded/og.jpg", "https://www.akiota.jp/uploaded/image/photo.jpg"]


# ---- 要約 ----


class _Meter:
    def __init__(self, seconds: list[float]) -> None:
        self.seconds_list = seconds

    def seconds(self, text: str) -> float:
        return self.seconds_list.pop(0)


class _LLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        self.calls.append(list(messages))
        return self.answers.pop(0)


NOW = datetime(2026, 10, 24, 6, 0, tzinfo=JST)
BODY = "町道津浪巡回線で、11月10日から11月14日まで終日全面通行止めを行います。迂回路をご利用ください。"


def test_summary_retry_once_on_length_then_ok() -> None:
    r = _record(title="通行止めのお知らせ", body=BODY, relevance="introduce")
    llm = _LLM(["町道津浪巡回線が通行止めです。", "町道津浪巡回線は、11月10日から11月14日まで終日全面通行止めです。迂回路をご利用ください。"])
    checker = news_summarize.make_checker("akiota", NOW)
    assert news_summarize.summarize_record(r, llm, _Meter([8.0, 17.0]), checker, CFG, NOW.date())  # type: ignore[arg-type]
    assert len(llm.calls) == 2 and "短すぎ" in llm.calls[1][-1]["content"]
    assert r.summary_checks["result"] == "ok" and r.status == "auto" and r.summary_checks["speech_sec"] == 17.0


def test_summary_held_on_ungrounded_number_and_length() -> None:
    r = _record(title="通行止めのお知らせ", body=BODY, relevance="introduce")
    # 本文に無い日付（11月20日）
    llm = _LLM(["町道津浪巡回線は、11月20日に終日全面通行止めです。迂回路をご利用ください。"])
    checker = news_summarize.make_checker("akiota", NOW)
    news_summarize.summarize_record(r, llm, _Meter([17.0]), checker, CFG, NOW.date())  # type: ignore[arg-type]
    assert r.summary_checks["result"] == "held" and r.status == "held"
    assert "本文に無い数・日付・支払方法" in r.summary_checks["reasons"]
    r2 = _record(title="通行止めのお知らせ", body=BODY, relevance="introduce")
    news_summarize.summarize_record(r2, _LLM(["通行止めです。", "通行止めです。"]), _Meter([5.0, 6.0]),  # type: ignore[arg-type]
                                    checker, CFG, NOW.date())
    assert "読み上げが短すぎる" in r2.summary_checks["reasons"]


def test_summary_drops_homepage_sentence() -> None:
    assert news_summarize.clean_summary("断水します。詳しくはホームページでご確認ください。", CFG) == "断水します。"


def test_staff_edited_summary_is_not_overwritten(tmp_path: Path) -> None:
    staff = "町道津浪巡回線は、11月10日から11月14日まで終日全面通行止めです。迂回路をご利用ください。"
    r = _record(title="通行止めのお知らせ", body=BODY, relevance="introduce", summary=staff, edited_by_staff=True,
                summarized_hash="old", content_hash="new", summary_checks={"result": "ok", "reasons": []})
    save_record(tmp_path, r)
    llm = _LLM([])  # 呼ばれたら pop で落ちる
    stats = news_summarize.summarize(tmp_path, "akiota", llm, None, CFG, now=NOW)  # type: ignore[arg-type]
    got = load_records(tmp_path)[0]
    assert got.summary == staff and got.summarized_hash == "new" and stats["staff"] == 1 and not llm.calls


def test_collect_while_site_down_then_recovered(tmp_path: Path) -> None:
    """市のサイトが止まっている間はキャッシュの本文で作り、戻った夜に一覧の掲載日と記事の画像が入る。"""
    import json as _json

    import build
    import httpx

    src = build.load_sources("etajima")
    src.rate_limit_sec = 0.0
    feed = "https://www.city.etajima.hiroshima.jp/cms/atom/news"
    art = "https://www.city.etajima.hiroshima.jp/cms/articles/show/12426"
    body = "<p>" + "11月10日から11月14日まで、町道の通行止めを行います。迂回路をご利用ください。" * 3 + "</p>"
    html = (f"<html><head><title>通行止めのお知らせ</title><meta property='og:image' content='/img/news.jpg'></head>"
            f"<body><h1>通行止めのお知らせ</h1><main>{body}</main></body></html>")
    up = {"ok": False}

    def handler(req: httpx.Request) -> httpx.Response:
        if not up["ok"] or req.url.path == "/robots.txt":
            return httpx.Response(500 if up["ok"] is False else 404)
        if str(req.url) == feed:
            return httpx.Response(200, headers={"content-type": "application/atom+xml"}, text=(
                f"<feed><entry><title>通行止めのお知らせ</title><link rel='alternate' href='{art}'/>"
                "<published>2026-10-20T09:00:00+09:00</published></entry></feed>"))
        if str(req.url) == art:
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        if req.url.path == "/img/news.jpg":
            return httpx.Response(200, headers={"content-type": "image/png"}, content=_png(800, 600))
        return httpx.Response(404)

    crawler = build.Crawler(src, tmp_path / "cache")
    crawler.client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    # 知識パックのキャッシュ（止まる前に取ったもの）: 一覧のリンクと、記事の抽出済みの本文
    title, text = build.extract_html(html, art)
    for url, links, t, x in ((feed, [art], "", ""), (art, [], title, text)):
        crawler._cache_path(url).write_text(_json.dumps({
            "url": url, "title": t, "text": x, "links": links, "breadcrumbs": [],
            "fetched_at": "2026-10-21T17:29:02+09:00"}, ensure_ascii=False), encoding="utf-8")
    store = tmp_path / "news"
    news_collect.collect(src, crawler, store, CFG, now=datetime(2026, 10, 24, 6, 0, tzinfo=JST))
    (rec,) = load_records(store)
    assert rec.published == "2026-10-21" and rec.image is None and not rec.image_checked  # 掲載日は取得日・画像なし
    up["ok"] = True
    stats = news_collect.collect(src, crawler, store, CFG, now=datetime(2026, 10, 25, 6, 0, tzinfo=JST))
    (rec,) = load_records(store)
    assert stats["unchanged"] == 1  # 本文は同じ（分類・要約はやり直さない）
    assert rec.published == "2026-10-20" and rec.image == f"{rec.id}.jpg" and rec.image_checked
    assert (store / rec.image).is_file()
