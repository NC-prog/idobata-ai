"""WP13 A: 区画の用途（use・use_rules・題名の規則）とパックの形式 2（ネットワークには繋がない）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import build


def _src(site: str = "akiota") -> build.Sources:
    return build.load_sources(site)


def test_decide_use_order() -> None:
    src = _src("akiota")
    soshiki = src.seeds[0]  # 組織（dialogue）
    rss = src.seeds[5]  # 町の新着（lookup）
    # 1. URL の規則（拠点）
    assert build.decide_use("https://www.akiota.jp/soshiki/8/123.html", "補助金のお知らせ", soshiki, src)[0] == "lookup"
    assert build.decide_use("https://www.akiota.jp/soshiki/list7-1.html", "町内求人情報", soshiki, src)[0] == "lookup"
    assert build.decide_use("https://www.akiota.jp/site/userguide/1.html", "本サイトについて", soshiki, src)[0] == "exclude"
    # 2. 題名の exclude（URL の規則が無いとき）
    assert build.decide_use("https://www.akiota.jp/soshiki/7/1.html", "令和9年度予算編成方針", soshiki, src)[0] == "exclude"
    # 3. 題名の引き上げ（lookup の区画・lookup の URL 規則のページは引き上げない／引き上げる）
    assert build.decide_use("https://www.akiota.jp/soshiki/10/9.html", "通行止めのお知らせ", rss, src)[0] == "dialogue"
    assert build.decide_use("https://www.akiota.jp/soshiki/10/9.html", "協調学習の取組", rss, src)[0] == "lookup"
    # exclude と引き上げが同時に当たれば exclude
    assert build.decide_use("https://www.akiota.jp/soshiki/10/9.html", "防災倉庫の入札", rss, src)[0] == "exclude"
    # 4. 区画の既定
    assert build.decide_use("https://www.akiota.jp/soshiki/7/2.html", "戸籍の届出", soshiki, src)[0] == "dialogue"
    # 区画の分からないページ（調べるで足したもの）は lookup
    assert build.decide_use("https://cs-akiota.or.jp/news/x/", "マルシェ", None, src)[0] == "lookup"


def test_title_lift_applies_to_url_rule_lookup_but_not_final() -> None:
    """URL の規則で lookup にした課でも、題名の引き上げ語の記事は dialogue。final の規則は題名の規則を見ない。"""
    src = _src("akiota")
    soshiki = src.seeds[0]
    assert build.decide_use("https://www.akiota.jp/soshiki/3/5.html", "新たな防災気象情報の運用", soshiki, src)[0] == "dialogue"
    assert build.decide_use("https://www.akiota.jp/soshiki/5/6.html", "もりカー乗合割引を開始します", soshiki, src)[0] == "dialogue"
    assert build.decide_use("https://www.akiota.jp/soshiki/3/7.html", "カスタマーハラスメント対策基本方針", soshiki, src)[0] == "lookup"
    assert build.decide_use("https://www.akiota.jp/soshiki/3/8.html", "令和9年度予算編成方針", soshiki, src)[0] == "exclude"
    use, why = build.decide_use("https://www.akiota.jp/soshiki/list7-1.html", "町内求人情報", soshiki, src)
    assert use == "lookup" and "確定" in why  # 題名の exclude（求人情報）より先


def test_seed_of_from_cache_follows_crawl_order(tmp_path: Path) -> None:
    src = _src("etajima")
    crawler = build.Crawler(src, tmp_path)
    page = build.Page("https://www.city.etajima.hiroshima.jp/cms/articles/show/1", "企業立地", "本文", [], "t",
                      ["トップページ > 組織で情報を探す > 商工観光課 > 企業立地"])
    seed = build.seed_of(page, crawler, src)
    assert seed is not None and seed.url.endswith("/224") and seed.use == "lookup"
    page2 = build.Page("https://www.city.etajima.hiroshima.jp/cms/articles/show/2", "住民票", "本文", [], "t",
                       ["トップページ > 暮らしのガイド > 届出・証明"])
    seed2 = build.seed_of(page2, crawler, src)
    assert seed2 is not None and seed2.use == "dialogue"
    lib = build.Page("https://www.library.etajima.hiroshima.jp/information", "利用案内", "本文", [], "t")
    seed3 = build.seed_of(lib, crawler, src)
    assert seed3 is not None and build.decide_use(lib.url, lib.title, seed3, src)[0] == "dialogue"
    crawler.close()


def test_chunks_carry_use_and_old_pack_reads_as_dialogue(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = build.Page("https://x/a", "A", "ごみの出し方について。" * 10, [], "t")
    p.use = "lookup"
    chunks = build.make_chunks([p])
    assert chunks and all(c.use == "lookup" for c in chunks)
    assert build.use_counts(chunks) == {"lookup": {"pages": 1, "chunks": len(chunks)}}
    # format 1（use の列が無い）のパックを読むと dialogue
    pq.write_table(pa.table({"id": ["c0"], "source_url": ["https://x/a"], "title": ["A"], "fetched_at": ["t"],
                             "text": ["本文"]}), tmp_path / "chunks.parquet")
    previous, _ = build.load_previous(tmp_path)
    assert [c.use for c in previous] == ["dialogue"]


def test_merge_daily_updates_use_without_reembedding() -> None:
    text = "ごみの出し方について。" * 10
    old = build.make_chunks([build.Page("https://x/a", "A", text, [], "t")])
    again = build.Page("https://x/a", "A", text, [], "t2")
    again.use = "lookup"
    merged, stats = build.merge_daily(old, [again])
    assert stats["unchanged"] == 1 and [c.use for c in merged] == ["lookup"]
    assert merged[0].embed_text() == old[0].embed_text()  # ベクトルは使い回せる


def test_bad_use_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    import pytest

    with pytest.raises(ValueError):
        build._seed({"url": "https://x/", "use": "talk"}, 0)
    with pytest.raises(ValueError):
        build._check_rules([{"pattern": "x"}], "テスト")


def test_title_words_are_regex_with_lookbehind() -> None:
    """「議会」は協議会・審議会に当たらない、「人事」は求人事業主に当たらない（ビルドの結果で見つかった誤り）。"""
    src = _src("akiota")
    soshiki = src.seeds[0]
    url = "https://www.akiota.jp/soshiki/7/1.html"
    assert build.decide_use(url, "安芸太田町社会福祉協議会", soshiki, src)[0] == "dialogue"
    assert build.decide_use(url, "上下水道料金審議会", soshiki, src)[0] == "dialogue"
    assert build.decide_use(url, "求人事業主の皆様へ", soshiki, src)[0] == "dialogue"
    assert build.decide_use(url, "議会だより", soshiki, src)[0] == "exclude"
    assert build.decide_use(url, "人事異動について", soshiki, src)[0] == "exclude"


def test_bus_word_is_narrowed_and_safety_words_lift() -> None:
    # 「バス」単独では上げない（観光バスツアー・企画チケット）。路線バス・バス停・身の安全の語は上げる（10/24 の判断）
    src = build.load_sources("akiota")
    soshiki = next(s for s in src.seeds if s.url == "https://www.akiota.jp/soshiki/")
    url = "https://www.akiota.jp/soshiki/8/1.html"
    assert build.decide_use(url, "江田島市観光バスツアー誘客促進事業補助金のご案内", soshiki, src)[0] == "lookup"
    assert build.decide_use(url, "広電バスMOBIRY TRAVEL企画チケット", soshiki, src)[0] == "lookup"
    assert build.decide_use(url, "「戸河内ICバスセンター」バス停位置の変更について", soshiki, src)[0] == "dialogue"
    assert build.decide_use(url, "バスの時刻表と運賃表", soshiki, src)[0] == "dialogue"
    assert build.decide_use(url, "不審なメールやメッセージにご注意ください", soshiki, src)[0] == "dialogue"
    assert build.decide_use(url, "山岳遭難を防ごう", soshiki, src)[0] == "dialogue"
