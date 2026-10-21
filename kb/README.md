# 知識パック（kb/）

窓口機はインターネットに出ない方針なので、自治体サイトのクロールと埋め込みの計算は
「ネットにつながる機体でのバッチ」で行い、成果物（1 ディレクトリ）をフロント／バックエンドに配る。
7560 の standalone では、同じ機体で作ってそのまま使う。

## 作り方

```bash
# リポジトリのルートで（kb/ は server/ とは別の uv 環境）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima              # 全部（15〜25 分）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima --weather-only   # 天気だけ（数秒）
```

- 出力は `$IDOBATA_DATA_DIR/kb/<site>/`（既定 `~/idobata-data/kb/<site>/`）。`kb/pack/<site>` はそこへのリンク（git 管理外）
- 作りかけを読ませないよう、別ディレクトリに作ってから差し替える。フロント（`server/`）は起動時に読み、天気は更新されたら読み直す
- 前回の `ETag` / `Last-Modified` を `$IDOBATA_DATA_DIR/kb/cache/<site>/` に覚えておき、変わっていないページは本文を取り直さない

| ファイル | 中身 |
|---|---|
| `manifest.json` | 生成日時、ページ数・チャンク数、埋め込みモデル名とプレフィックス、取得の統計、ソース |
| `chunks.parquet` | `id, source_url, title, fetched_at, text`（出典の提示に使う） |
| `index.lancedb/` | 埋め込みベクトル（LanceDB。サーバプロセスは立てない） |
| `weather.json` | 気象庁の府県天気予報（今日・明日）。天気の文言は気象庁のまま |

## ソース定義（`kb/sources/<site>.yaml`、人間が編集する）

- `allow_domains`: 取得してよいドメイン
- `seeds`: 起点 URL・深さ・採用する URL（`include`）・辿らない URL（`exclude`）
- `exclude_breadcrumb_keywords` / `exclude_title_keywords`: **個人名を含みうるページ（議会・会議録・入札・人事・職員採用・首長の部屋・選挙）を外す**。知識パックに人名を入れない
- `weather`: 気象庁の府県予報区（`area_code`）・一次細分区域（`area_sub_code`）・気温の地点（`point_name`）
- `crawl`: robots.txt を守る、同じドメインへは `rate_limit_sec` 秒に 1 回、`max_pages`（採用ページ数）と `max_fetches`（取得数）の上限

## 中身の作り方

1. 取得: robots.txt を守り、1 秒 1 リクエスト。HTML だけ（PDF は第 2 段階で広報紙と一緒に扱う）
2. 抽出: `trafilatura` で本文を Markdown にする（ナビ・フッタを落とす）
3. チャンク: 見出し単位を基本に 400〜800 字、長い節は 100 字重ねて切る。同じ文面は 1 つにまとめる
4. 埋め込み: **Ruri v3（`cl-nagoya/ruri-v3-30m`、Apache-2.0）を CPU で**。文書は「検索文書: タイトル\n本文」、
   検索時は「検索クエリ: 発話」（モデルカードの 1+3 プレフィックス規約どおり）
5. 検索（`server/src/knowledge.py`）: ベクトル検索と文字 2-gram の BM25 を RRF で融合し、
   ベクトルの類似度がしきい値未満のものを落として上位 5 件・合計 1,500 字まで

## 夜間バッチ

`deploy/standalone/kb-nightly.{service,timer}`（毎晩 2 時に全部）と `kb-weather.{service,timer}`
（5 時・11 時・17 時の発表の後に天気だけ）は systemd の user ユニットの雛形。sudo は要らない。
有効化は人間が判断する:

```bash
cp deploy/standalone/kb-{nightly,weather}.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now kb-nightly.timer kb-weather.timer
loginctl enable-linger "$USER"    # ログアウト中も動かす場合（sudo 不要の範囲で、ポリシーによる）
```

site プロファイルでは、作った `~/idobata-data/kb/<site>/` を rsync で現地機に配る前提（配布スクリプトは未作成）。
