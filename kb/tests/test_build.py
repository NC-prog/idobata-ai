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


def test_weather_keeps_jma_wording_and_uses_amedas() -> None:
    raw = json.loads((FIXTURES / "jma_340000_20261021.json").read_text(encoding="utf-8"))

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("latest_time.txt"):
            return httpx.Response(200, text="2026-10-21T15:40:00+09:00")
        if "/amedas/data/map/20261021154000.json" in req.url.path:
            return httpx.Response(200, json={"67511": {"temp": [23.0, 0]}})
        return httpx.Response(200, json=raw)

    src = build.load_sources("etajima")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        w: dict[str, Any] = build.build_weather(src, client)
    today, tomorrow = w["forecasts"]
    assert today["weather"] == raw[0]["timeSeries"][0]["areas"][0]["weathers"][0]  # 文言はそのまま
    assert today["region"] == "広島県南部"
    assert today["pops"][0] == {"from": 12, "to": 18, "pop": 70}
    assert "max" not in today  # 予報の代表点（広島）の気温は使わない
    assert w["observation"] == {"station_id": "67511", "station": "呉", "place": "江田島",
                                "in_site": False, "temp": 23.0, "observed_at": "2026-10-21T15:40+09:00"}
    assert w["source"] == "気象庁（広島地方気象台）"


def test_amedas_missing_value_is_skipped() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("latest_time.txt"):
            return httpx.Response(200, text="2026-10-21T15:40:00+09:00")
        return httpx.Response(200, json={"67212": {"temp": [None, 5]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert build.build_observation(
            {"station_id": "67212", "station_name": "加計", "place_name": "安芸太田"}, client
        ) is None


def test_redact_names() -> None:
    r = build.redact_names
    assert r("▶MC\n\n本名 正憲 ／ 小杉 怜子\n") == "▶MC\n\n（氏名省略）\n"
    assert r("江田島市長　山田　太郎") == "江田島市長　（氏名省略）"
    assert r("副院長（兼）看護部長 松原　花子") == "副院長（兼）看護部長 （氏名省略）"
    assert r("考案者 伊藤奈々") == "考案者 （氏名省略）"
    assert r("講師は山田 太郎さんです。") == "講師は（氏名省略）さんです。"
    assert r("支部長 木村 隆（きむら たかし）") == "支部長 （氏名省略）"
    assert r("移住してきた山田 一郎さん・花子さんご一家") == "移住してきた（氏名省略）さんご一家"
    assert r("ＭＣは小杉怜子さん、本名正憲さん") == "ＭＣは（氏名省略）さん、（氏名省略）さん"
    assert r("浜田省吾さんの聖地") == "（氏名省略）さんの聖地"
    assert r("- 三﨑　花子さん") == "- （氏名省略）さん"
    # 人名でないもの（課名・宛先・規則の文）は残す
    for keep in ["江田島市役所 市民課 宛", "市長が必要と認める書類", "江田島市福祉保健部　保健医療課　宛",
                 "教育長および教育委員は、町長が議会の同意を得て任命します。",
                 "代表者の氏名を書いてください", "全身が撮影され、データ処理により",
                 "皆さんのご協力", "社協さんへ", "お客様各位", "ご家族の皆様",
                 "ご利用者様", "民生委員さん", "参加事業者さん", "猪肉屋さん", "様々な行事"]:
        assert r(keep) == keep


def test_versioned_pack_swap(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    root.mkdir()
    legacy = root / "etajima"
    legacy.mkdir()
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")
    cur = build.current_pack(root, "etajima")  # 古い実体ディレクトリは版に移ってリンクになる
    assert (root / "etajima").is_symlink() and (cur / "manifest.json").is_file()
    new = build.versions_dir(root, "etajima") / "20990101000000"
    new.mkdir()
    build.swap_link(root / "etajima", new)
    assert (root / "etajima").resolve() == new.resolve()
    extra = build.versions_dir(root, "etajima") / "20000101000000"
    extra.mkdir()
    build.prune_versions(root, "etajima")
    assert len(list(build.versions_dir(root, "etajima").iterdir())) == build.KEEP_VERSIONS
    assert new.exists()


def test_clean_title() -> None:
    assert build.clean_title("特産品ショップ「えたじまーれ」&nbsp;| 買う | 江田島市観光協会【公式】") == "特産品ショップ「えたじまーれ」"
    assert build.clean_title("住民票などの窓口発行") == "住民票などの窓口発行"


def test_merge_daily_reuses_unchanged_pages() -> None:
    text_a = "ごみの出し方について。" * 10
    text_b = "秋祭りを開催します。" * 10
    pa_ = build.Page("https://x/a", "A", text_a, [], "t")
    pb = build.Page("https://x/b", "B", text_b, [], "t")
    previous = build.make_chunks([pa_, pb])
    # A は同じ、B は本文が変わった、C は新しい、D は短すぎてチャンクにならない
    pb2 = build.Page("https://x/b", "B", "花火大会は10月31日です。" * 10, [], "t2")
    pc = build.Page("https://x/c", "C", "図書館の休館日のお知らせです。" * 10, [], "t2")
    pd = build.Page("https://x/d", "D", "短い", [], "t2")
    a_again = build.Page("https://x/a", "A", text_a, [], "t2")
    merged, stats = build.merge_daily(previous, [a_again, pb2, pc, pd])
    assert stats == {"unchanged": 1, "changed": 1, "new": 1, "no_chunks": 1}
    urls = [c.source_url for c in merged]
    assert urls.count("https://x/a") == 1 and "https://x/c" in urls and "https://x/d" not in urls
    assert next(c for c in merged if c.source_url == "https://x/a") is previous[0]  # 使い回し
    assert "花火大会" in next(c for c in merged if c.source_url == "https://x/b").text


def test_embed_reuse_skips_unchanged_text(tmp_path: Path, monkeypatch: Any) -> None:
    chunks = build.make_chunks([build.Page("https://x/a", "A", "ごみの出し方について。" * 10, [], "t")])
    reuse = {build.embed_key(c.embed_text()): [0.1] * 4 for c in chunks}
    import sys
    import types

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("使い回せるのに埋め込もうとした")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=boom))
    stats = build.embed_and_save(chunks, tmp_path, reuse)
    assert stats == {"embedded": 0, "reused": len(chunks)}
