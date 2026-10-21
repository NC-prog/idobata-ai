# 知識パック（kb/）

知識パックの更新のための取得（市・町のサイト、観光協会、気象庁など、`sources/<site>.yaml` の
`allow_domains` にあるドメインだけ）は、窓口機から直接行ってよい（2026-10-21 の方針改定、`docs/構想メモ.md` 5 章）。
会話・音声・映像は外に出さない。7560 の standalone では、同じ機体で作ってそのまま使う。

## 作り方

```bash
# リポジトリのルートで（kb/ は server/ とは別の uv 環境）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima              # 全部（15〜25 分）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima --weather-only   # 天気だけ（数秒）
```

- 出力は `$IDOBATA_DATA_DIR/kb/<site>`（既定 `~/idobata-data/kb/<site>`）。**これはシンボリックリンク**で、実体は
  `<site>.versions/<日時>/`。新しい版を別ディレクトリに作り終えてからリンクを付け替える（無停止差し替え）。直前の版は残す
- フロント（`server/`）は `manifest.json` の変化を 60 秒ごとに見て、再起動なしで新しい版に切り替える。天気（`weather.json`）は更新されたら読み直す
- `kb/pack/<site>` はそこへのリンク（git 管理外）
- `--from-cache`: 取得し直さず、今のパックのページをキャッシュから作り直す（チャンク分け・人名除去の規則を変えたとき）
- 前回の `ETag` / `Last-Modified` を `$IDOBATA_DATA_DIR/kb/cache/<site>/` に覚えておき、変わっていないページは本文を取り直さない

| ファイル | 中身 |
|---|---|
| `manifest.json` | 生成日時、ページ数・チャンク数、埋め込みモデル名とプレフィックス、取得の統計、ソース |
| `chunks.parquet` | `id, source_url, title, fetched_at, text`（出典の提示に使う） |
| `index.lancedb/` | 埋め込みベクトル（LanceDB。サーバプロセスは立てない） |
| `weather.json` | 気象庁の府県天気予報（今日・明日の天気と降水確率。文言は気象庁のまま）と、アメダス最寄り観測点の気温の実況 |

## ソース定義（`kb/sources/<site>.yaml`、人間が編集する）

- `allow_domains`: 取得してよいドメイン
- `seeds`: 起点 URL・深さ・採用する URL（`include`、正規表現）・辿らない URL（`exclude`、正規表現）。ほかに次を書ける
  - `scope_breadcrumbs`: パンくずにこの語を含むページだけ採用・展開する
  - `sitemap: true`: 起点が sitemap.xml（索引なら子も読む）。sitemap の `lastmod` がキャッシュより古いページは取り直さない
  - `render: playwright`: ブラウザで描画してから本文を取る（JS で描画するサイト。江田島市観光協会）
  - `require_text`: 本文にこの語を含むページだけ採用する（県の観光サイトから拠点の分だけ、など）
  - `max_fetches`: その seed での取得の上限
  - `recheck_days`: キャッシュがこの日数より新しければ取り直さない（大きなサイトを数日かけて回す）
- `exclude_breadcrumb_keywords` / `exclude_title_keywords`: **個人名を含みうるページ（議会・会議録・入札・人事・職員採用・首長の部屋・選挙）を外す**。知識パックに人名を入れない
- `weather`: 気象庁の府県予報区（`area_code`）・一次細分区域（`area_sub_code`）と、気温を取るアメダスの観測点（`amedas`）
  - 観測点の ID は気象庁のアメダス地点一覧（`https://www.jma.go.jp/bosai/amedas/const/amedastable.json`）で確かめて書く
  - 江田島は呉（67511。島内に気温の観測点が無い）、安芸太田は加計（67212）
  - 予報の気温の代表点（広島・庄原）は拠点から遠いので使わない
- `default_trust` / `default_label`: 出どころの信頼度と、応答で出典を言うときの言い方（seed ごとに `trust` / `label` で上書き）
  - trust は `official`（市町・県）/ `semi`（観光協会・社協・道の駅など）/ `news`
  - 応答の最後の「詳しくは〇〇にも載っています」の〇〇は、使った資料の label から決まる
- `crawl`: robots.txt を守る、同じドメインへは `rate_limit_sec` 秒に 1 回、`max_pages`（採用ページ数）と `max_fetches`（取得数）の上限

## 中身の作り方

1. 取得: robots.txt を守り、1 秒 1 リクエスト。HTML だけ（PDF は第 2 段階で広報紙と一緒に扱う）
2. 抽出: `trafilatura` で本文を Markdown にする（ナビ・フッタを落とす）
3. チャンク: 見出し単位を基本に 400〜800 字、長い節は 100 字重ねて切る。同じ文面は 1 つにまとめる
4. 埋め込み: **Ruri v3（`cl-nagoya/ruri-v3-30m`、Apache-2.0）を CPU で**。文書は「検索文書: タイトル\n本文」、
   検索時は「検索クエリ: 発話」（モデルカードの 1+3 プレフィックス規約どおり）
5. 検索（`server/src/knowledge.py`）: ベクトル検索と文字 2-gram の BM25 を RRF で融合し、
   ベクトルの類似度がしきい値未満のものを落として上位 5 件・合計 1,500 字まで

## 自動更新（既定で有効）

systemd の user タイマー（sudo 不要）で、**毎朝 6:00 に全体**、**毎時 05 分に天気**を更新する。

```bash
bash deploy/standalone/install-kb-timers.sh            # .env の IDOBATA_SITE（既定 etajima）
bash deploy/standalone/install-kb-timers.sh akiota     # 別の拠点も
systemctl --user list-timers 'kb-*'                    # 次の実行時刻
journalctl --user -u 'kb-nightly@etajima' -n 50        # ログ
systemctl --user disable --now kb-nightly@etajima.timer kb-weather@etajima.timer   # 止める
```

- ユニットは `deploy/standalone/kb-{nightly,weather}@.{service,timer}`（`%i` が拠点）。install スクリプトが uv とリポジトリのパスを埋めて `~/.config/systemd/user/` に置く
- 全体と天気は `flock` で重ならない（全体の作り直し中の天気は飛ばす）
- ログアウト中や再起動直後（ログイン前）も動かすには linger が要る。スクリプトは `loginctl enable-linger` を試し、できなければ `sudo loginctl enable-linger <ユーザー>` を人間に頼むよう表示する（7560 では sudo なしで有効にできた）

site プロファイルでは、作った `~/idobata-data/kb/<site>/` を rsync で現地機に配る前提（配布スクリプトは未作成）。
