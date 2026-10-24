# 知識パック（kb/）

知識パックの更新のための取得（市・町のサイト、観光協会、気象庁など、`sources/<site>.yaml` の
`allow_domains` にあるドメインだけ）は、窓口機から直接行ってよい（2026-10-21 の方針改定、`docs/構想メモ.md` 5 章）。
会話・音声・映像は外に出さない。7560 の standalone では、同じ機体で作ってそのまま使う。

## 作り方

```bash
# リポジトリのルートで（kb/ は server/ とは別の uv 環境）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima              # 全部（40〜60 分。毎週日曜 3:00）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima --daily      # 更新の速い区画だけ（数分。毎朝 6:00）
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site etajima --weather-only   # 天気だけ（数秒）
```

- 出力は `$IDOBATA_DATA_DIR/kb/<site>`（既定 `~/idobata-data/kb/<site>`）。**これはシンボリックリンク**で、実体は
  `<site>.versions/<日時>/`。新しい版を別ディレクトリに作り終えてからリンクを付け替える（無停止差し替え）。直前の版は残す
- フロント（`server/`）は `manifest.json` の変化を 60 秒ごとに見て、再起動なしで新しい版に切り替える。天気（`weather.json`）は更新されたら読み直す
- `kb/pack/<site>` はそこへのリンク（git 管理外）
- `--from-cache`: 取得し直さず、今のパックのページをキャッシュから作り直す（チャンク分け・人名除去の規則を変えたとき）
- `--from-cache-pack <版のディレクトリ>`（複数可）: `--from-cache` で、今のパックに加えてその版のページも組み直す
  （区画や use の規則を変えて、今の版から落ちたページを前の版から戻すとき。WP13）
- `--daily`: `daily: true` の seed（お知らせ・イベント・広報・防災・休館日）だけを**深さ 1** で取り、今の版に差分で足す
  - 取ったページの本文ハッシュ（`page_hash`）が前の版と同じなら、そのチャンクとベクトルをそのまま使う
  - 変わったページ・新しいページだけを作り直す。取らなかったページはそのまま残る（消えたページは毎週の全体で落ちる）
  - 変わったページが無ければ版を作らない
- **埋め込みの使い回し**: 全体の作り直しでも、埋め込む文（タイトル + 本文）が前の版と同じチャンクはベクトルを使い回し、
  変わったものだけ埋め込む。ログの「ベクトル: 新しく埋め込み N 件 / 使い回し M 件」で確かめられる
- 前回の `ETag` / `Last-Modified` を `$IDOBATA_DATA_DIR/kb/cache/<site>/` に覚えておき、変わっていないページは本文を取り直さない

| ファイル | 中身 |
|---|---|
| `manifest.json` | 形式（`format: 2` から `use` を持つ）、生成日時、ページ数・チャンク数・用途ごとのページ数（`use_counts`）、埋め込みモデル名とプレフィックス、取得の統計、ソース |
| `chunks.parquet` | `id, source_url, title, fetched_at, text, trust, label, page_hash, use`（出典の提示と、会話で使う範囲の絞り込みに使う） |
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
  - `daily: true`: 更新の速い区画。毎朝の `--daily` ではこの seed だけを深さ 1 で取る
- `use`（seed ごと）: そのページを会話で使うか（WP13。「集めるのは広く、会話で使うのは狭く」）。書かなければ `dialogue`
  - `dialogue`: 毎ターンの検索で会話の参考資料にしてよい（暮らしのガイド・課の窓口・防災・時刻表など）
  - `lookup`: 「調べる」（来訪者が同意したとき）と夜回しのときだけ検索する（観光・広報・お知らせの混在区画・事業者向け）
  - `exclude`: 集めない（パックに入れない）
  - `why`: その use にした理由（分類表に載る）
- `use_rules`（seed ごと・拠点ごと）: URL の正規表現でページごとに use を変える。`{pattern, use, why, final}` の並び。
  seed の規則 → 拠点の規則の順に見て、最初に当たったものを使う
  - `final: true` の規則はその use で**確定**（題名の規則より先。手続きナレッジの出典・求人の一覧・サイトの管理ページ）
  - `final` でない規則は「既定」を差し替えるだけ（課の区画を lookup にしても、題名の引き上げ語の記事は dialogue に戻る）
- `kb/scope_rules.yaml`（全拠点に効く、題名の語での use の規則）:
  - `exclude_title_words`: 題名に当たれば exclude（求人情報・入札・議会・例規・統計・人事・公告・予算・決算・監査・パブリックコメント）
  - `dialogue_title_words`: 既定が lookup の題名に当たれば dialogue に引き上げる（防災・ごみ・休館・路線バス・バス停・断水・支所・予防接種・詐欺・不審など。「バス」単独は観光の記事も上げるので使わない）
  - 語は正規表現として当てる（「協議会」「審議会」を議会にしないために `(?<![協審評])議会` と書く）
  - 決め方の順: URL の規則（final）→ 題名の exclude → 題名の dialogue 引き上げ → 既定（final でない URL の規則 → seed の use）。
    exclude と dialogue の語が同時に当たれば exclude
- 古い形式のパック（`use` 列が無い）は、全部 `dialogue` として読む（フロントも差分の組み直しも）
- 「調べる」の夜回しで足したページは `lookup`
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

## お知らせ（WP11）

`sources/<site>.yaml` の `news`（新着・お知らせ・イベントの Atom / RSS）から、住民に広く関係する記事を集めて夜のうちに分類と要約を作る。
フロントは来訪者の同意を取って 1 会話 1 件紹介する（`docs/WP11_result.md`）。

```bash
CUDA_VISIBLE_DEVICES= uv run --project kb python kb/build.py --site akiota --news-only       # お知らせだけ（収集 → 分類 → 要約）
uv run --project kb python kb/news_classify.py --site akiota --force                         # 分類だけやり直す（news.yaml の語を変えたとき）
uv run --project kb python kb/news_summarize.py --site akiota --force                        # 要約だけ作り直す（職員が直したものは除く）
```

- `--daily` と全体の作り直しの両方で、知識パックの取得の前に動く（`--no-news` で外す）。失敗してもパックの更新は続ける
- 記事は `$IDOBATA_DATA_DIR/kb/<site>.news/<id>.json`（と画像 `<id>.jpg`）。パックの各版に `news` というリンクを置く（版を差し替えても残る）
- 分類の語と分類 9 つ・重みは `server/context/news.yaml`。規則で決まらないものだけ LLM（7560 の Qwen3-8B）。要約は LLM と VOICEVOX を使い、
  保存の前に読み上げ前の安全装置（server の dialogue.py / news.py）に通す。kb の環境から `server/` を import する
- 画像は記事と同じホスト（許可ドメイン内）の og:image か本文の最初の画像で、幅 400px 以上のものだけ。長辺 1280 に縮める
- 取れないとき（市のサイトのメンテナンスなど）は前回取った内容で続ける

## 自動更新（既定で有効）

systemd の user タイマー（sudo 不要）で更新する:

| タイマー | いつ | 中身 |
|---|---|---|
| `kb-daily@<拠点>` | 毎朝 6:00 | `--daily`（更新の速い区画だけ、数分。江田島で約 185 リクエスト・4〜5 分） |
| `kb-full@<拠点>` | 毎週日曜 3:00 | 全体の作り直し（40〜60 分。変わっていないチャンクは再埋め込みしない） |
| `kb-weather@<拠点>` | 毎時 05 分 | 天気だけ（数秒） |

```bash
bash deploy/standalone/install-kb-timers.sh            # .env の IDOBATA_SITE（既定 etajima）
bash deploy/standalone/install-kb-timers.sh akiota     # 別の拠点も
systemctl --user list-timers 'kb-*'                    # 次の実行時刻
journalctl --user -u 'kb-daily@etajima' -n 50          # ログ
systemctl --user disable --now kb-daily@etajima.timer kb-full@etajima.timer kb-weather@etajima.timer   # 止める
```

- ユニットは `deploy/standalone/kb-{daily,full,weather}@.{service,timer}`（`%i` が拠点）。install スクリプトが uv とリポジトリのパスを埋めて `~/.config/systemd/user/` に置く（以前の `kb-nightly@` は外す）
- 毎朝・全体・天気は `flock` で重ならない（作り直し中の天気は飛ばす）
- ログアウト中や再起動直後（ログイン前）も動かすには linger が要る。スクリプトは `loginctl enable-linger` を試し、できなければ `sudo loginctl enable-linger <ユーザー>` を人間に頼むよう表示する（7560 では sudo なしで有効にできた）

site プロファイルでは、作った `~/idobata-data/kb/<site>/` を rsync で現地機に配る前提（配布スクリプトは未作成）。
