# WP13 結果 — 会話で使う知識の絞り込み（集めるのは広く、会話で使うのは狭く）

対象機: 7560（standalone、Qwen3-8B-AWQ）。ブランチ `feature/wp13-scope`（main 00e3293 から）。指示書は `docs/WP13_knowledge_scope.md`。

先に 1 コミット（9941b0c）: 意地悪 20 番の案。聞き直しで課の名前の文を「同じ文の繰り返し」として捨て、残りに課の名前が無くなったら、
捨てた課の名前の文を 1 つ言い直す（`test_repeated_question_keeps_department_name`）。

止まる条件の確認:
- 30 問の正答: main 27・27、WP13（最終）28・26・27。途中の版で 1 回 4 問下がった（25・23）ので、原因（課の名前の省略・観光の型分け）を直した。use の付け方では戻らない種類だった
- パックの形式変更: `check_procedures.py` は全チャンク（use を問わず）を読むので壊れない。「調べる」の夜回しで足したページは lookup で入り、
  「調べる」は dialogue と lookup の両方を探す。古い形式（use 列なし）は全部 dialogue として読む（フロント・差分の組み直しとも、テストあり）
- 型分け: 手続き 100 問で procedure から外れたのは 2 問（観光案内の 2 問。観光の型にしたもので、意図どおり）。B の初版では 9 問外れ、語の追加で 0 にした

## A. 区画の用途（use）

- 区画（seed）に `use: dialogue | lookup | exclude` と `why`、区画と拠点に URL の `use_rules`（`pattern`・`use`・`why`・`final`）。全拠点の題名の規則は `kb/scope_rules.yaml`
- 決め方の順（10/24 の指示の 1）: **URL の規則（final: true）→ 題名の exclude → 題名の dialogue 引き上げ → 既定**（final でない URL の規則 → 区画の use）。
  指示の「URL の規則が先」をそのまま全部に当てると、総務課・企画 DX 課などを URL の規則で lookup にしたとき、題名の引き上げ（防災気象情報・ライドシェア）が効かない。
  そこで URL の規則は「既定を差し替える」だけにし、確定させたいもの（手続きナレッジの出典・求人の一覧・サイトの管理ページ）だけ `final: true` にした
- 題名の語は正規表現（`(?<![協審評])議会`・`(?<!求)人事`。「社会福祉協議会」「審議会」「求人事業主」を外さない）。exclude と引き上げの語が両方当たれば exclude
- パックの形式 2: `chunks.parquet` と LanceDB に `use` 列、`manifest.json` に `format: 2` と `use_counts`。exclude のページはパックに入れない
- `--from-cache-pack <版>`: キャッシュから組み直すときに前の版のページも戻す（江田島市のサイトがメンテナンス中で HTTP 500 だったので、江田島はキャッシュから組んだ）
- 迷い印（🔶）は 10/24 に確定したもの: 江田島の時刻表（124）は dialogue。atom/news・topics、安芸太田の RSS 3 本は lookup（題名の引き上げ語の記事は dialogue）。
  商工観光課・会計課・産業観光課・企画 DX 課・総務課・農業委員会・筒賀支所・/site/kyoiku/・移住・ふるさと納税・morica は lookup。夜回しで足したページは lookup

### 分類表（江田島）

| 区画 | URL | use | 理由 |
|---|---|---|---|
| > 暮らしのガイド | https://www.city.etajima.hiroshima.jp/cms/categories/show/1 | dialogue |  |
| 組織で情報を探す > 市民課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/213 | dialogue |  |
| 組織で情報を探す > 税務課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/215 | dialogue |  |
| 組織で情報を探す > 生活環境課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/217 | dialogue |  |
| 組織で情報を探す > 社会福祉課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/218 | dialogue |  |
| 組織で情報を探す > 保健医療課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/221 | dialogue |  |
| 組織で情報を探す > 高齢介護課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/499 | dialogue |  |
| 組織で情報を探す > 子育て支援課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/222 | dialogue |  |
| 組織で情報を探す > 商工観光課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/224 | lookup | 観光・企業立地・求人が中心で、窓口の手続きはほぼ無い（WP13。題名の引き上げ語の記事は dialogue） |
| 組織で情報を探す > 会計課 | https://www.city.etajima.hiroshima.jp/cms/categories/show/232 | lookup | 支払・口座の事業者向けが多い（WP13） |
| 各課・公共施設のご案内 | https://www.city.etajima.hiroshima.jp/cms/categories/show/22 | dialogue |  |
| 各種相談窓口 | https://www.city.etajima.hiroshima.jp/cms/categories/show/128 | dialogue |  |
| 時刻表 | https://www.city.etajima.hiroshima.jp/cms/categories/show/124 | dialogue |  |
|  | https://www.city.etajima.hiroshima.jp/cms/atom/events | dialogue |  |
|  | https://www.city.etajima.hiroshima.jp/cms/calendars | dialogue |  |
|  | https://www.city.etajima.hiroshima.jp/cms/atom/news | lookup | 入札・人事から暮らしまで混在。急ぎの案内は WP11 のお知らせが受け持つ（WP13） |
|  | https://www.city.etajima.hiroshima.jp/cms/atom/topics | lookup | 同上（WP13） |
| > 防災 | https://www.city.etajima.hiroshima.jp/cms/categories/show/11 | dialogue |  |
| 広報えたじま | https://www.city.etajima.hiroshima.jp/cms/categories/show/29 | lookup | 広報紙（WP13） |
| 観光協会のホームページ | https://etajima-kanko.com/sitemap.xml | lookup | 観光（WP13） |
| 観光協会のホームページ | https://etajima-kanko.com/sitemap.xml | lookup | 観光（WP13） |
| 社会福祉協議会のホームページ | https://etajima-syakyo.org/sitemap.xml | lookup | 事例・職員募集・予算決算・規約は会話に要らない。相談窓口のページだけ dialogue（WP13）。URL の規則: `^https://etajima-syakyo\.org/pages/(23|24|29|31|32|46|47|48|54|55|69)/?$` → dialogue |
| 社会福祉協議会のホームページ | https://etajima-syakyo.org/pages/57?detail=1&b_id=727&view=list&block727_limit=20 | lookup | ブログ的な記事（WP13） |
| 市立図書館のホームページ | https://www.library.etajima.hiroshima.jp/sitemap.xml | lookup | 新着・行事は「調べる」で。利用案内・開館は dialogue（WP13）。URL の規則: `^https://www\.library\.etajima\.hiroshima\.jp/(information|faq|minilibrary)?/?$` → dialogue |
| 県の観光サイト | https://dive-hiroshima.com/sitemap.xml | lookup | 観光（WP13） |

### 分類表（安芸太田）

| 区画 | URL | use | 理由 |
|---|---|---|---|
|  | https://www.akiota.jp/soshiki/ | dialogue |  |
|  | https://www.akiota.jp/calendar/ | dialogue |  |
| あきおおたからのホームページ | https://cs-akiota.or.jp/news/category/イベント/ | lookup | 観光・道の駅（WP13） |
| あきおおたからのホームページ | https://cs-akiota.or.jp/special/category/観光/ | lookup | 観光・道の駅（WP13） |
|  | https://www.akiota.jp/soshiki/14/ | dialogue |  |
|  | https://www.akiota.jp/rss/10/list1.xml | lookup | 混在（10/24 の「協調学習」はここから）。急ぎの案内は WP11 のお知らせ。題名の引き上げ語の記事は dialogue（WP13） |
|  | https://www.akiota.jp/rss/10/list3.xml | lookup | 混在（10/24 の「協調学習」はここから）。急ぎの案内は WP11 のお知らせ。題名の引き上げ語の記事は dialogue（WP13） |
|  | https://www.akiota.jp/rss/10/list6.xml | lookup | 混在（10/24 の「協調学習」はここから）。急ぎの案内は WP11 のお知らせ。題名の引き上げ語の記事は dialogue（WP13） |
| 道の駅のお知らせ | https://cs-akiota.or.jp/special/category/道の駅/ | lookup | 観光・道の駅（WP13） |
| 社会福祉協議会のホームページ | https://www.akiota-shakyo.jp/news/feed/ | lookup | ブログ的な記事（WP13） |
| 社会福祉協議会のホームページ | https://www.akiota-shakyo.jp/wp-sitemap.xml | dialogue | 。URL の規則: `/(news|event)(/|$)` → lookup |
| 県の観光サイト | https://dive-hiroshima.com/sitemap.xml | lookup | 観光（WP13） |
|  | https://www.akiota.jp/life/1/ | dialogue |  |
|  | https://www.akiota.jp/life/2/ | dialogue |  |
| （拠点の URL の規則） | `/site/kyoiku/(8264|10011)\.html$` | dialogue（確定） | 手続きナレッジの出典（就学援助・黎明館） |
| （拠点の URL の規則） | `/soshiki/list7-` | lookup（確定） | 町内求人情報の一覧（「求人がある」ことは言える。題名の exclude より先） |
| （拠点の URL の規則） | `/site/userguide/|/soshiki/list[0-9]` | exclude（確定） | サイトの使い方・新着一覧の一覧ページ |
| （拠点の URL の規則） | `/soshiki/8/` | lookup | 産業観光課（補助金・観光・職業紹介。職業紹介所は手続きナレッジの就労で答える） |
| （拠点の URL の規則） | `/soshiki/25/` | lookup | 道の駅推進チーム |
| （拠点の URL の規則） | `/soshiki/5/` | lookup | 企画DX課（計画・SNS。ライドシェア・もりカーは題名の引き上げで dialogue） |
| （拠点の URL の規則） | `/soshiki/3/` | lookup | 総務課（指定管理・計画・方針。防災気象情報は題名の引き上げで dialogue） |
| （拠点の URL の規則） | `/soshiki/(18|22)/` | lookup | 農業委員会・筒賀支所（委員名簿・プロジェクト。支所の窓口案内は題名の引き上げで dialogue） |
| （拠点の URL の規則） | `/site/(ijyu|furusatonouzei|profile|kouhou|bbikou|morica)/` | lookup | 移住・空き家、ふるさと納税、町のプロフィール、広報、ブロードバンド、地域通貨 |
| （拠点の URL の規則） | `/site/kyoiku/` | lookup | 大会・事業・プロポーザルが多い |
| （拠点の URL の規則） | `/life/3/` | lookup | 事業者向け |

### ページ数・チャンク数の前後

| 拠点 | 前（形式 1） | 後（形式 2） | dialogue | lookup | exclude（入れない） |
|---|---|---|---|---|---|
| 江田島 | 704 ページ・1175 チャンク | 694 ページ・1150 チャンク | 514 ページ・927 チャンク | 180 ページ・223 チャンク | 10 ページ（パブリックコメント・予算・求人情報） |
| 安芸太田 | 925 ページ・1653 チャンク | 799 ページ・1457 チャンク | 364 ページ・695 チャンク | 435 ページ・762 チャンク | 125 ページ（userguide・新着一覧の一覧ページ・予算・公告・求人情報） |

- 安芸太田で exclude 以外に消えたのは 1 ページ（「企画DX課」の課の紹介ページ。URL の規則 /soshiki/5/ の区画の取り方で落ちた）
- 手続きナレッジの出典: 新しいパックから消えたのは江田島の 1 件（下書き shuro の出典の求人記事 706。求人情報で exclude）。lookup になったのは 11 件（江田島 1・安芸太田 10）。
  手続きナレッジは毎ターンのパックの検索とは別に引くので、会話には影響しない

### 題名の引き上げで dialogue になった記事

- 江田島（2 件）: 江田島市観光バスツアー誘客促進事業補助金のご案内 / 浜田省吾さんゆかりのバス停、展示室へお越しの皆様へ
- 安芸太田（18 件）: 定額ライドシェアの本格運行を開始します！ / 広電バスMOBIRY TRAVEL企画チケット2026秋「秋の三段峡あるき」 / 広島電鉄バスMOBIRY TRAVEL企画チケット【夏のさとやま探し】 /
  広電バスMOBIRY TRAVEL「安芸太田町おでかけバスきっぷ【春のさとやま遊び】」 / もりカー乗合割引を開始します！ / 【道の駅再整備事業】工事区域及びバス停の位置の変更について /
  「戸河内ICバスセンター」バス停位置の変更について / 新たな防災気象情報の運用が始まりました / moricaで税(料)金及び窓口手数料の支払いができるようになりました /
  おたふくかぜ予防接種助成事業 / 筒賀支所の宿直業務廃止のお知らせ / 新・公共交通「もりカー」運行開始！ / 安芸太田町の防災マップ（2023） / 災害時の避難方法 /
  林野火災注意報・林野火災警報の運用開始について / 安芸太田町地域防災計画について / 安芸太田町防災情報メール通知サービス / バスの時刻表と運賃表
- 🔶「バス」で観光の記事（観光バスツアーの補助金・MOBIRY の企画チケット 3 件・浜田省吾のバス停）も上がる。観光バスツアーの補助金は 30 問の「おすすめの観光地」に付いた
  （下の C。型分けを直して付かなくなった）。「バス」を「路線バス・バス停・時刻表」に狭めるかは判断待ち

### lookup に落ちた暮らし寄りの記事（江田島、区画の既定で lookup。市のサイトの lookup は 51 件で、大半は宿・カフェ・観光の記事）

なくそう海の事故 ～ルールを守って楽しく遊泳しましょう～ / 山岳遭難を防ごう ～体力・経験に応じた安全な登山計画を～ / ふれあいプラザさくら ～営業日変更のお知らせ～ /
職場のマタハラ（マタニティハラスメント）でつらい思いをしていませんか？ / 平成３０年７月豪雨災害義援金 ・ 寄附金 ・ 支援金の受付終了について / 支払金の口座振込依頼手続きについて /
不審なメールやメッセージにご注意ください / フリーマーケットの品物を受け付けています！ / オータムフェスタ江田島2026・江田島湾海上花火大会を開催します！ /
全国の島々が集まる祭典！「アイランダー2026」開催！ / 【広島県×御船印コラボ】ひろしま瀬戸内 船めぐりデジタル・スタンプラリー / 海上自衛隊課業整列見学一般募集について /
ご利用ください！ レンタサイクル【市観光協会】 / 竹炭を稲作に活用しませんか？ / えたじま飲食店マップを作成しました！ / 広報えたじま第263号（令和８年１０月号） /
ドクターイエロー展示計画に関する市長コメント / サンビーチおきみ・やすらぎ交流農園の売却と新ホテル事業計画について / 【受付終了】江田島市誘客促進イベント実施支援補助金のご案内 /
「えたじま観光アンバサダー」二次募集を開始！

- 🔶 花火大会・アイランダーなどの行事は、行事の型（event）でお知らせ（WP11）の紹介する記事から渡る。海の事故・山岳遭難・不審なメール・マタハラの相談は引き上げ語に「事故」「遭難」「詐欺・不審」「相談」を足すかが判断待ち

## B. 発話の型と、型ごとの検索の範囲

- `server/context/intents.yaml`（語）と `server/src/intents.py`（`classify_intent`・`nouns`）。優先 procedure > facility > event > tourism > weather > other > smalltalk。
  procedure は語に加えて、別名表（procedures.yaml）と手続きナレッジ（承認済み・下書き）の name・aliases で当たる。課の担当業務（handles）の語は見ない
  （商工観光課の「観光」で観光の問いが手続きになる）
- C の評価で直した: 下書き「観光案内」の別名「観光」「観光地」のように、**観光の語を含む名前・別名だけで当たったときは procedure にしない**（「江田島でおすすめの観光地」が procedure になり、
  観光バスツアーの補助金が付いていた）
- `KnowledgeConfig.scope_by_intent`（`IntentScope`）:

| 型 | 渡す資料 |
|---|---|
| procedure | 承認済みの手続きナレッジ＋固定事実。パックは渡さない。承認済みが無いときだけ dialogue の上位 2 ページ（同じページのチャンクは 1 件と数える） |
| facility | 固定事実だけ |
| event | dialogue の上位 3 とお知らせ（WP11）の紹介する記事。名詞の足切りあり（「今週末」は具体的な日付にして当てる） |
| tourism | 渡さない。「調べてみましょうか？」の提案を優先（lookup はそのときに探す） |
| weather | 天気だけ |
| other | dialogue の上位 3。名詞の足切りあり |
| smalltalk | 渡さない |

- 名詞の足切り（B-2）: 発話の名詞（カタカナ 2 字以上・漢字 2 字以上・漢字 1 字＋送り仮名。助詞で始まる仮名は続けない）のどれかが、資料の題名か本文の先頭 200 字にあること
- 承認済みの手続きで答えたのに応答に課の名前が無ければ、規則で決まった担当の課を「担当は市民課（1階）です。」と一文足す（`ProcedureConfig.approved_department_sentence`。C の評価で直した）
- 毎ターンのログ: 「型: procedure / 渡した資料: 承認済み 1・パック 0・お知らせ 0」
- 「調べる」（lookup.py）: パック（dialogue と lookup）を先に探し、名詞の当たるものが無ければ許可ドメインへ
- テスト: `server/tests/test_wp13_unit.py`（型分け 35 例・優先・名詞・承認済みでパックなし・雑談/場所/観光/天気は検索しない・足切り・行事にお知らせ・観光で調べるの提案・
  調べるがパック先行・観光の別名・課の一文・ページで 2 件）、`kb/tests/test_scope.py`（決め方の順・引き上げ・final・区画の見当て・use 列・差分の組み直し）

## C. 評価

同じ LLM（7560 の Qwen3-8B-AWQ）・同じ時刻帯（2026-10-24 の夕方〜夜）で、main（00e3293 の worktree と前のパック）と WP13 を続けて回した。

### 資料・脱線・音声開始（計測。問いごとに会話を区切り、渡した資料の題名と最初の文までの時間を記録）

| 集合 | ターン | 資料（パック・お知らせ）を渡したターン main → WP13 | 脱線の候補 main → WP13 | 最初の文まで（中央値・秒） main → WP13 |
|---|---|---|---|---|
| 評価セット 30 問 | 32 | 18 → 3 | 5 → 1 | 0.49 → 0.33 |
| 雑談 10 問 | 10 | 4 → 0 | 0 → 0 | 0.11 → 0.09 |
| 意地悪 30 問（複数ターンはターンごと） | 40 | 5 → 0 | 1 → 0 | 0.50 → 0.26 |
| 手続き 100 問 | 100 | 95 → 32 | 41 → 14 | 0.59 → 0.36 |

- 脱線の候補（補助の判定）: 渡した資料（手続きナレッジ以外）の題名に発話の名詞が無く、題名の型が発話の型と違うもの。指示の主の定義（題名が発話と無関係）は人の目で見た
- WP13 の候補 15 件を見た: 発話と関係のある資料が 11 件（税の納付→コンビニ納付・PayB、ごみ→ごみ出しカレンダー、家の補助金→空き家対策の補助、農業→農地法の許可、
  食べ物→えたじまん焼き など）。**無関係と判断したのは 4 件**: ふるさと納税→国税の相談・義援金（丁寧）、ふるさと納税→税証明書の郵送（くだけた）、
  店をやりたい・補助→ふれあいサロン（社協のサロン。これで社会福祉課と言った＝手続き 98 番）、公民館の部屋→図書館の「利用案内」。
  どれも承認済みの手続きが無い手続きの問い（dialogue の上位 2）で、名詞の足切りは指示どおり other / event だけに掛けている
- main の候補は 41 件のうち、他人事の事例ブログ（社協の「お皿の片付け」「しおかぜネット」）・犬の LINE を死亡届に・百歳体操を死亡後の手続きに・百歳体操を成人式に、など
  WP13 で消えたもの
- 音声開始: 資料が短くなった分、最初の文まで 0.1〜0.25 秒速い（プロンプトの参考資料はターンごとに変わり、プレフィックスキャッシュが効かない部分）

### 正答

| 集合 | main | WP13 |
|---|---|---|
| 評価セット 30 問（補助判定 OK、32 問） | 27・27（2 回） | 25・23 → 直して 28・26・27（3 回） |
| 雑談 10 問（OK・締めの定型で終わる） | 9・1/10 | 9・1/10 |
| 意地悪 30 問（OK） | 29（20 番: 5 回目に保健医療課なし） | 30（先の 1 コミット） |
| 手続き 100 問 | 課の誤り 1・作り話 0・言い回し 0・課を言わない 12・担当の課の案内が無い 1 | 課の誤り 1・作り話 1・言い回し 1・課を言わない 2・担当の課の案内が無い 0 |

- 30 問の落ちの中身: main は 3・16（市民課を言わない）と締めの定型。WP13 の最初の版は 3・14・16・27（市民課を言わない）が増えた。
  承認済みの手続きだけを渡すと、8B が場所の一覧（本庁・市民センター・支所…）だけを言って課の名前を省く。**承認済みの手続きの応答に課の名前が無ければ規則で決まった課の一文を足す**
  ようにして、課の名前の落ちは 0〜1 に戻った（残り 19 番「明日」は天気の言い方の揺れ、ほかは締めの定型で、どちらも資料の範囲とは関係しない）
- 「江田島でおすすめの観光地」が procedure（下書き「観光案内」の別名「観光」）になり、引き上げ語「バス」で dialogue になった観光バスツアーの補助金を
  「観光地のご案内」として話していた（補助判定は OK だったが脱線）。観光の語の別名だけでは procedure にしないようにした
- 手続き 100 問の「課を言わない」12 → 2 は、課の一文を足したため。課の誤り 98 番（店・補助→社会福祉課）は WP13 で 4 回とも出る（上の脱線 1 件）。
  96 番（イノシシ→市民課）は上位 2 件が同じページの 2 チャンクで埋まっていたので、ページで 2 件にして直った。作り話 68 番は今日の日付（10月24日）を言ったもので、評価の判定の誤り
- 評価の作り話の判定が、承認済みの「１４日以内」（全角）と応答の「14日以内」を別の字として数えていた。NFKC で照らすようにした（main の数字も直した判定で回し直したもの）

### 10/24 の 4 つの脱線（安芸太田、`server/tests/eval/cases_regression_1024.yaml`）

| 流れ | main | WP13 |
|---|---|---|
| r1 国保→国保税 | 脱線あり（国保税の Q&A が付き「国保税についてご相談ですね」） | なし（承認済み「国民健康保険の加入」だけ） |
| r2 児童手当→求人の通勤手当 | なし（WP11 の後始末で消えていた） | なし（資料なし） |
| r3「わかりました」→協調学習 | なし（同上） | なし（雑談の型で資料なし） |
| r4 町から案内→道の駅のブログ | なし（同上） | なし（お知らせの紹介に入る） |

脱線した流れ: main 1 / 4 → WP13 0 / 4（3 回とも 0）。

### 判断待ち（Norio）

- 引き上げ語「バス」で観光の記事（観光バスツアーの補助金・MOBIRY の企画チケット・浜田省吾のバス停）も dialogue になる。「路線バス」「バス停」「時刻表」に狭めるか
- 江田島の lookup に落ちた暮らし寄りの記事（海の事故・山岳遭難・不審なメール・マタハラの相談・ふれあいプラザさくらの営業日）に引き上げ語を足すか
- 承認済みの無い手続きの問い（dialogue の上位 2）に名詞の足切りを掛けるか（指示は other / event だけ。ふれあいサロンは題名・本文の先頭に「補助」が無いので落ちる。ほかは未確認）

## D. 文書

- CLAUDE.md: 「集めるのは広く、会話で使うのは狭く」の段落と WP13 の行
- `kb/README.md`: `use`・`why`・`use_rules`（`final`）・`scope_rules.yaml`・`--from-cache-pack`・形式 2
- `docs/残タスク.md` 2-3 の備考: 「WP13 で資料の雑音を減らした。残りは T2 で比べる」

## ついでに直したもの

- お知らせの置き場の `.fetch_status.json` を記事として読んで警告を出していた（pathlib の glob は「.」で始まる名前にも当たる）。除外した（dc928cd）
