"""kb/build.py の純粋な関数のテスト（ネットワークには繋がない）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build
import httpx

FIXTURES = Path(__file__).parent / "fixtures"


def test_chunk_sizes_and_overlap() -> None:
    para = "住民票の写しは市民課で交付しています。" * 10  # 190 字
    md = "## 住民票\n" + para + "\n## 印鑑登録\n" + para + "\n## 長い節\n" + ("あいうえお。" * 300)
    chunks = build.chunk_text(md)
    assert all(len(c) <= build.CHUNK_MAX for c in chunks)
    assert all(len(c) >= build.CHUNK_DROP_BELOW for c in chunks)
    start = next(i for i, c in enumerate(chunks) if "長い節" in c)
    long_parts = chunks[start:]
    assert len(long_parts) >= 2
    # 長い節は重ねて切る（次のチャンクの先頭は前のチャンクの末尾と重なる）
    assert long_parts[1][:50] in long_parts[0]


def test_short_fragments_dropped_and_duplicates_merged() -> None:
    p1 = build.Page("https://x/1", "A", "## ナビ\nトップへ", [], "t")
    text = "ごみの出し方について。" * 10
    p2 = build.Page("https://x/2", "B", text, [], "t")
    p3 = build.Page("https://x/3", "C", text, [], "t")
    chunks = build.make_chunks([p1, p2, p3])
    assert [c.source_url for c in chunks] == ["https://x/2"]


def test_breadcrumbs_and_exclusion() -> None:
    html = ("<ol><li><a>トップページ</a></li>\n<li>&gt; <a>組織で情報を探す</a></li>\n"
            "<li>&gt; 議会事務局</li></ol><p>本文</p>")
    crumbs = build.extract_breadcrumbs(html)
    assert crumbs and "議会事務局" in crumbs[0]
    src = build.load_sources("etajima")
    page = build.Page("https://x", "お知らせ", "本文", [], "t", crumbs)
    assert build._excluded(page, src)
    assert build._excluded(build.Page("https://x", "令和7年第3回定例会会議録", "本文", [], "t"), src)


def test_seed_scope_and_regex() -> None:
    seed = build.Seed(url="u", include=[r"/cms/articles/show/\d+$"],
                      exclude=[r"/cms/categories/(show|articles)/(14|175)$"],
                      scope_breadcrumbs=["> 暮らしのガイド"])
    assert seed.matches(seed.include, "https://e/cms/articles/show/123")
    assert not seed.matches(seed.include, "https://e/cms/articles/show/123/print")
    assert seed.matches(seed.exclude, "https://e/cms/categories/show/14")
    assert not seed.matches(seed.exclude, "https://e/cms/categories/show/1")
    assert seed.in_scope(build.Page("u", "", "", [], "t", ["トップページ > 暮らしのガイド > 税金"]))
    assert not seed.in_scope(build.Page("u", "", "", [], "t", ["トップページ > 市政情報"]))


def test_weather_keeps_jma_wording() -> None:
    raw = json.loads((FIXTURES / "jma_340000_20261021.json").read_text(encoding="utf-8"))

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=raw)

    src = build.load_sources("etajima")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        w: dict[str, Any] = build.build_weather(src, client)
    today, tomorrow = w["forecasts"]
    assert today["weather"] == raw[0]["timeSeries"][0]["areas"][0]["weathers"][0]  # 文言はそのまま
    assert today["region"] == "広島県南部" and today["max"] == "28" and today["min"] == ""
    assert tomorrow["max"] == "24" and tomorrow["min"] == "15"
    assert w["source"] == "気象庁（広島地方気象台）"


def test_redact_names() -> None:
    r = build.redact_names
    assert r("▶MC\n\n本名 正憲 ／ 小杉 怜子\n") == "▶MC\n\n（氏名省略）\n"
    assert r("江田島市長　山田　太郎") == "江田島市長　（氏名省略）"
    assert r("副院長（兼）看護部長 松原　花子") == "副院長（兼）看護部長 （氏名省略）"
    assert r("考案者 伊藤奈々") == "考案者 （氏名省略）"
    assert r("講師は山田 太郎さんです。") == "講師は（氏名省略）さんです。"
    assert r("支部長 木村 隆（きむら たかし）") == "支部長 （氏名省略）"
    assert r("移住してきた山田 一郎さん・花子さんご一家") == "移住してきた（氏名省略）さんご一家"
    # 人名でないもの（課名・宛先・規則の文）は残す
    for keep in ["江田島市役所 市民課 宛", "市長が必要と認める書類", "江田島市福祉保健部　保健医療課　宛",
                 "教育長および教育委員は、町長が議会の同意を得て任命します。",
                 "代表者の氏名を書いてください", "全身が撮影され、データ処理により"]:
        assert r(keep) == keep
