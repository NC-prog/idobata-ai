# WP01 結果報告 — 応答の速さ（A）／エッジ分離＋ブラウザアバター（B）

ブランチ: `feature/wp01-edge-avatar`（main へのマージは人間判断）
機体: 7560（standalone、ASR・LLM・VOICEVOX コンテナ稼働）

---

## A. 応答の速さ

実施: 2026-10-21

### 結論（A）

- **発話終了 → 音声開始は 0.63〜0.99 秒**（疑似音声 3 本 × 2 回、WP00 は 1.95〜2.89 秒）。完了条件の 1.3 秒未満を満たす
- **「能美」は「ノウミ」と読まれる**ようになった（VOICEVOX の解析結果と、合成音声の ASR 書き起こしで確認）
- **実マイクでの 3 往復は未確認（人間の確認待ち）**。理由は次の 2 つ
  - 声を出せる人がいない
  - 内蔵マイク（hw:0,0）が無音時でもフルスケールの雑音を返すようになっていた。WP00 の最終確認では使えていた。詳細は A-4
- 速くするために、指示書にない変更を 2 つ入れた
  1. 1 文目だけ読点でも切る
  2. VOICEVOX の合成スレッド数を 8 にする
  
  どちらも下の「決めたこと」に書いた

### A-1. 文単位の先行合成

| 箇所 | 変更 |
|---|---|
| `llm_client.py` | `chat_stream()` を追加（`stream=True`、`stream_options.include_usage`）。最初の断片を受け取る前の失敗は `chat()` と同じくリトライする。途中で切れた場合はそのまま例外にする（言いかけの文を言い直さない） |
| `dialogue.py` | `respond()` を非同期ジェネレータにした。`SentenceSplitter` で「。」「？」「！」（全角・半角）と改行で文に切る。文末記号が続く場合（「？！」）や閉じ括弧（「。」」）は同じ文に含める。英語の "." は直後が空白のときだけ切る。履歴には、言い終えた全文を 1 ターンとして積む。途中で失敗したら、言えた文だけを積む |
| `tts_client.py` | `synthesize_with_query()`（WAV と audio_query を返す。B の口パク用）。`postPhonemeLength=0.1`。VOICEVOX への呼び出しは接続失敗・5xx を 2 回まで再試行する。`StreamPlayer` を追加した（下記） |
| `speech_output.py`（新規） | 出力先の抽象。`LocalSpeaker`（StreamPlayer）と `WavDirWriter`（1 応答ぶんの文を 1 つの WAV に繋いで書き出す。`--output-wav-dir` 用）。B で `EdgeSpeaker` を足す |
| `main.py` | 1 ターンの流れ: LLM の文生成（タスク）→ 文キュー → 直列に合成 → 出力先へ push。ゲートは 1 文目を流す直前に `mute()`。push のたびに `mute(expected_sec=再生終了までの見込み)` で安全装置を張り直す。最後の文を鳴らし終えたら `schedule_unmute()`。途中では開けない |

`StreamPlayer`: 文ごとの WAV を **1 本の OutputStream** に続けて書く。次の文の合成が終わっていない間は 20ms の無音を書いて繋ぐ。文ごとにストリームを開き直すと途切れ、ALSA ではアンダーランの雑音にもなるため、この方式にした。実機の内蔵スピーカーで「経過 12.30 秒 / 音声 12.34 秒」「14.92 秒 / 14.89 秒」となり、音声の長さどおりに鳴り終わった（余計な隙間なし）。

`[計測]` の表示（例）:

```
[計測] 発話終了→ ASR 0.25s / LLM 1文目確定 0.50s / 音声開始 0.81s / LLM 完了 1.09s
```

### 計測結果（疑似音声 `--input-wav` 3 本、VAD の発話終了判定が起点）

| 往復 | 回 | ASR | LLM 1文目確定 | **音声開始** | LLM 完了 |
|---|---|---|---|---|---|
| 1 挨拶 | 1 | 0.26 | 0.51 | **0.76** | 1.23 |
| 2 手続き | 1 | 0.35 | 0.52 | **0.99** | 1.58 |
| 3 雑談 | 1 | 0.31 | 0.41 | **0.73** | 1.19 |
| 1 挨拶 | 2 | 0.25 | 0.50 | **0.81** | 1.09 |
| 2 手続き | 2 | 0.35 | 0.52 | **0.92** | 1.27 |
| 3 雑談 | 2 | 0.32 | 0.42 | **0.63** | 1.48 |

- `--output-wav-dir` での値（1 文目の合成完了 = 音声開始）。内蔵スピーカーへの実再生（`--input-wav` + `--output-device`）でも 0.65〜1.00 秒だった
- 変更を順に入れたときの音声開始（同じ 3 本）:
  - 文で切るだけ: 0.84 / 1.62 / 1.44 秒
  - 1 文目を読点でも切る: 0.72 / 1.37 / 0.69 秒
  - VOICEVOX を 8 スレッドに: 上の表
- 実体感はこれに VAD の無音判定 0.4 秒が加わる

### A-2. 固有名詞の読み（VOICEVOX ユーザー辞書）

- `server/context/<site>.yaml` の語を、文字列のほかに `{surface, reading, accent}` の辞書で書けるようにした（`src/site_context.py`）。ASR のバイアシングは surface を使う。YAML は 1 つのまま
- 起動時に `TTSClient.sync_user_dict()` が `GET /user_dict` と突き合わせる。同じ表記が無ければ `POST /user_dict_word`、読み・アクセント・優先度が違えば `PUT /user_dict_word/{uuid}`、同じなら何もしない。VOICEVOX に繋がらなければ警告して、既定の読みのまま続ける
- **優先度は 9**（`VoicevoxConfig.user_dict_priority`）。既定の 5（7 でも同じ）ではシステム辞書に負けて「能美」が「ノミ」のままだった。9 以上で効いた（VOICEVOX 0.25.2 で確認）
- 登録した語:

| 拠点 | 表記 | 読み | accent | 登録前の VOICEVOX |
|---|---|---|---|---|
| 江田島 | 江田島 | エタジマ | 0 | エタジマ（正） |
| | 能美 | ノウミ | 0 | **ノミ** |
| | 能美島 | ノウミジマ | 0 | **ノミトオ** |
| | 大柿 | オオガキ | 0 | オオガキ（正） |
| | 沖美 | オキミ | 0 | **オキヨシ** |
| | 小用 | コヨウ | 0 | **ショオヨオ**（小用港はコヨオコオで正） |
| | 切串 | キリクシ | 2 | キリクシ（正） |
| | 中町 | ナカマチ | 2 | ナカマチ（正） |
| 安芸太田 | 安芸太田 | アキオオタ | 0 | ア'キ／オオタ（2 語に分かれる） |
| | 加計 | カケ | 0 | カケ（正） |
| | 戸河内 | トゴウチ | 2 | トゴオチ（正） |
| | 筒賀 | ツツガ | 0 | ツツガ（正） |
| | 三段峡 | サンダンキョウ | 0 | サンダンキョオ（正） |
| | 温井 | ヌクイ | 0 | ヌクイ（正） |

- **accent について**: 指示書の既定値（省略時 1）を全語に当てると、すべて頭高（サ'ンダンキョウ、ナ'カマチ）になり、もともと正しかった語のアクセントが変わってしまう。そこで YAML に accent を明示した
  - 読みが元から正しかった語: VOICEVOX 自身の解析と同じアクセント（文中「〜に行きます」で一致することを確認）
  - 読みを直した 5 語（能美・能美島・沖美・小用・安芸太田）: 平板（0）とした。**正しいかは耳での確認が要る**
  - コードの既定値は指示どおり 1（`user_dict_default_accent`）
- 「能美の秋祭りはいつですか」（text モード）の確認:
  - 応答「能美の秋祭りの日程について、詳しく教えていただけますか？…」の audio_query は `ノウミノ'/アキマ'ツリノ/…`
  - 合成した WAV を Qwen3-ASR で書き起こすと、語彙なしで「**の海**の秋まつり」、拠点語彙ありで「**能見**の秋祭り」。どちらも「のうみ」と読まれている
  - WP00.5 の登録前は「蚤（のみ）」だった
- 統合テスト `test_user_dict_fixes_place_name_reading` を追加した（`uv run pytest -m integration`）

### A-3. 完了条件の状況

| 条件 | 結果 |
|---|---|
| `--mode voice --audio-device 0 --output-device 20` で 3 往復、音声開始 1.3 秒未満 | **疑似音声では OK**（0.63〜0.99 秒、内蔵スピーカーへの実再生でも OK）。**実マイクは未実施**（A-4） |
| `--mode text` で「能美の秋祭りはいつですか」→「のうみ」 | **代替確認 OK**（上記）。耳での確認は人間 |
| pytest / mypy strict / ruff | OK（下の「品質」） |

### A-4. 実機の音声デバイスで見つかったこと（人間に確認してほしい）

1. **Avaya B109（USB スピーカーフォン）が挿さっていて、sounddevice のデバイス番号が 2 つずれていた**
   - 内蔵スピーカーは 20 → **21**。20 は今「Default Source」（入力）
   - 指示書どおり `--output-device 20` を付けると、出力デバイスを開けずに既定デバイスへフォールバックし、そこで再生が戻らなくなる（安全装置で 20 秒後に閉じる）
   - 既定デバイス（PipeWire の ALSA プラグイン）で再生が戻らないのは、従来の `play()` でも同じ。WP00 の `sd.play()` が戻らなかった件と同根と思われる
   - **対策として、`--audio-device` / `--output-device` / `.env` でデバイスを名前の一部でも指定できるようにした**。例: `--audio-device hw:0,0 --output-device alsa_output.pci-0000_00_1f.3`。番号がずれても壊れない
2. **内蔵マイク（hw:0,0、device 0）が無音時に RMS 0.6〜1.0 のフルスケール雑音になっていた**
   - WP00 の最終確認（10-19）では使えていた
   - ALSA ミキサーは Capture 63/63（+30dB）、Headset Mic / Headphone Mic は off、Internal Mic の項目は無し
   - ミキサーは変更していない（システム設定のため）
   - PipeWire 経由（alsa_input…analog-stereo）も同じ雑音
3. **Avaya B109 のマイク（device 9、16kHz）は正常**（無音時 RMS 0.004）
   - ただし内蔵スピーカーから鳴らした来訪者役の音声はほとんど拾わなかった（スピーカーフォン側のエコー対策か、置き場所のため）
   - そのため、内蔵スピーカー → マイクの音響ループバックでの代替確認はできなかった
   - **実マイクの 3 往復は、人間が Avaya B109（`--audio-device 9` か名前 `Avaya`）か、B のブラウザマイクで行ってほしい**

### 決めたこと（A）

- **1 文目だけ読点でも切る**（`first_chunk_soft_terminators="、，,"`、読点より前が 4 文字以上のとき）
  - 1 文目が長いと（「住民票の写しは市区町村の役所で取り寄せられます。」）、その合成待ちで音声開始が 1.4〜1.6 秒になったため
  - 「はい、」だけを先に言わないよう、4 文字未満では切らない
  - 2 文目以降は、前の文の再生中に合成できるので句点でだけ切る
  - 読点の位置は VOICEVOX でももともと息継ぎが入るので、0.2 秒（前後の無音）の間は不自然になりにくい
- **VOICEVOX の合成スレッド数を 8 に**（`docker-compose.yaml` に `VV_CPU_NUM_THREADS=${VOICEVOX_CPU_THREADS:-8}`）
  - 別ポートに一時コンテナを立てて実測した。17 字の文: 既定 689ms / 4 スレッド 736ms / **8 スレッド 458ms** / 16 スレッド 481ms
  - VOICEVOX コンテナだけを作り直した（ASR・LLM は止めていない）
  - 作り直すとユーザー辞書は消えるが、アプリが起動時に登録し直すので問題ない
- 文と文の間は `postPhonemeLength=0.1` + 次の文の `prePhonemeLength`（既定 0.1）で約 0.2 秒

### 人間が確認すること（A）

1. **実マイクで 3 往復**（Avaya B109 か B のブラウザ）。`[計測]` の音声開始が 1.3 秒未満か
2. **内蔵マイクの雑音**: WP00 から何が変わったか（Avaya を挿したこと、ミキサー、PipeWire の設定）。デモでは Avaya B109 を使う前提にするか
3. **アクセント**（耳で）: 能美・能美島・沖美・小用・安芸太田を平板（0）にした。例: 「能美の秋祭り」「沖美町」「小用港」「安芸太田町役場」
4. **読みが確信できず登録していない語**（YAML にある語。括弧内は Claude Code の推定と、現在の VOICEVOX の読み）:
   - 安芸太田
     - 龍頭峡（推定 リュウズキョウ。VOICEVOX は **ヨンドオカイ** で誤り）
     - 道の駅来夢とごうち（推定 ライムトゴウチ。VOICEVOX は **ライユメトゴウチ**）
     - 上殿（推定 カミトノ。VOICEVOX は カミドノ）
     - 殿賀（トノガ）、坪野（ツボノ）、津浪（ツナミ）、井仁（イニ）
   - 江田島
     - 三高港（推定 ミタカコウ。VOICEVOX は **サンコウミナト**）
     - さとうみ科学館（VOICEVOX は サト／ウミカガクカン と 2 語に分かれる）
   
   正しい読みが分かれば、YAML に `{surface, reading, accent}` で書き足すだけで登録される
5. 応答の 2 往復目で LLM が「市区町村の役所で」と答えた（江田島市役所のロビーなのに一般論）。ペルソナに拠点名を入れるのは WP03（対話 FSM）で扱う想定

### 品質（A の時点）

- `uv run pytest`: 155 passed（integration 2 件は別に `-m integration` で 2 passed）
- `uv run mypy --strict src`: Success（16 files）
- `uv run ruff check`: All checks passed

---

## B. エッジ分離＋ブラウザアバター

実施: 2026-10-21

### 結論（B）

- **`--mode voice --edge` で、ブラウザをエッジにした 3 往復が通った**
  - ブラウザのマイクで聞き取り、応答はブラウザで再生、口は音素に合わせて動く
  - `playback_ended` の後にゲートが開く
- 確認は**ヘッドレス Chrome** で行った。マイクの代わりに、来訪者役の WAV を偽のマイクとして流した（下記）
  - 音声開始は **0.79 / 1.07 / 0.97 秒**（1.3 秒未満）
- **人が画面の前で実マイクを使った確認はまだ**。マイクは A-4 の内蔵マイク雑音の件があるので、Avaya B109 を選ぶ必要がありそう
- ローカル再生モード（`--edge` なし）も引き続き動く（A の確認を再実行して 0.69〜1.41 秒。下記「計測」）

### B-1. エッジ ⇄ フロントの契約（実装したもの）

- WebSocket は 1 本（`ws://127.0.0.1:8765`）で、JSON（テキスト）とバイナリを混在させる。ページは同じプロセスの FastAPI から `http://localhost:8080/edge/` で配信する
- 指示書の表どおり: `hello` / PCM16 バイナリ / `playback_ended`（上り）、`state` / `caption` / `speak`＋WAV / `theme`（下り）。`interrupt` は受けてログに出すだけ
- **契約に 1 つ足した: `playback_ended` に `last_seq`（最後に鳴らし終えた文の番号）**
  - 次の文の合成が間に合わないと、ブラウザの再生キューは文と文の間で一瞬空になる
  - `utterance_id` だけでは、この途中の通知と「応答の最後まで鳴らした」通知を見分けられない
  - フロントは `last_seq` が最後に送った文の番号以上になるまで待つ
- 口パク `mouth`（`src/lipsync.py`）: audio_query の `prePhonemeLength` から始めて、各モーラの子音長・母音長を積み上げる。母音 a/i/u/e/o を `aa/ih/ou/ee/oh` に写す
  - N・cl・句読点の無音は閉口なので出さない
  - 無声化した母音（「です」の「ス」）も出さない
  - 統合テストで、最後の口形区間が WAV の長さ（文末の無音を除く）に収まることを確認した
- 状態の遷移（フロント側で送る）:
  - VAD の発話開始 → `listening`
  - 発話終了 → `thinking`（ASR が空なら `idle`）
  - 1 文目を送るとき → `speaking`
  - `playback_ended`（最後の文）→ `idle`
- 字幕:
  - 来訪者: ASR 確定時に `caption{role:user, final:true}`
  - アバター: 文を送るたびに、そこまでの全文を `caption{role:assistant}`
  - ページ側では、アバターの字幕を「その文が鳴り始めたとき」に出す（届いた時点で先に出すと、音声より先に字幕が読めてしまうため）
- `audio_input.EdgeAudioInput`: 20ms（320 サンプル）単位で届く PCM16 を、VAD の 512 サンプルに詰め直す。VAD 以降のパイプラインは変更なし
- `--edge` のときは `EdgeSpeaker` が `speak`＋WAV を送り、`playback_ended` を待ってからゲートを開く
  - 通知が来なくても、再生終了の見込み + 3 秒で開く
  - `--edge` なしなら、従来どおりローカル再生（`LocalSpeaker`）
- `hello` を受けたら `theme{site, season}`（季節は月から決める）と `state:idle` を送る

### B-2. エッジページ（`edge/`）

- Vite 8 + TypeScript 7、three.js 0.186 + @pixiv/three-vrm 3.5
- `npm run build`（`tsc --noEmit` → `vite build`）の成果物は `server/static/edge/` に出る（git 管理外）
- マイク: `getUserMedia({audio:{echoCancellation, noiseSuppression, autoGainControl: true}})` で取る
  - 16kHz の AudioContext を作り、ブラウザ側でリサンプルする
  - AudioWorklet（`public/mic-processor.js`）で PCM16 にして、20ms ごとに送る
  - 送信が詰まったら（bufferedAmount 64KB 超）古い音声を溜めずに捨てる
  - `hello.caps.aec` には、トラックが実際に AEC 有効になったか（`getSettings().echoCancellation`）を入れる
- 再生: 届いた WAV を `decodeAudioData` し、前の文の終わりに続けて `start()` する（隙間なし）
  - デコードは届いた順に直列で行う
  - キューが空になったら `playback_ended` を送る
- アバター（`src/avatar.ts`）:
  - 腕を下ろした姿勢で、胸から上を写す
  - idle: 呼吸（spine の小さな揺れ）と、3〜6 秒おきのまばたき
  - listening: 少し前傾して、相手（カメラ）を見る
  - thinking: 視線と顔を少し外す
  - speaking: 口パク（viseme を 1/28 秒程度で追従）と軽い頷き
  - 表情は happy / relaxed を 0.1〜0.25 だけ乗せる
- 字幕: 画面下に 2 行（来訪者 = 黄色、アバター = 白）。文字は画面高の 3.4〜4.2%（1080p で 37〜45px）
- 背景: `theme` で `public/backgrounds/<site>_<season>.jpg` に切り替える
  - 今回は単色グラデーションのプレースホルダ（Pillow で生成、2 拠点 × 4 季節）。実写は市町から受け取って差し替える
- 起動の流れ:
  - 通常は「はじめる」ボタン（AudioContext の自動再生制限とマイク許可のため、1 回だけ押す）
  - 自動再生が許可された起動（キオスク用フラグ `--autoplay-policy=no-user-gesture-required`）なら、ボタンなしで始まる
  - WebSocket が切れたら 0.5〜5 秒間隔で張り直す

### 決めたこと（B）

- **VRM モデル**: `VRM1_Constraint_Twist_Sample.vrm`。three-vrm リポジトリの examples に同梱されている唯一のモデル
  - モデル内のライセンス設定（VRMC_vrm.meta）を読んで確認した。推測ではない
  - `edge/public/models/README.md` に転記した

  | 項目 | 値 |
  |---|---|
  | 配布元 | github.com/pixiv/three-vrm `packages/three-vrm/examples/models/`（dev ブランチ、2026-10-21 取得） |
  | 作者 / 著作権 | pixiv Inc. / (c) 2022 pixiv Inc. |
  | ライセンス | VRM パブリック・ライセンス 1.0（https://vrm.dev/licenses/1.0/） |
  | avatarPermission | everyone（ライセンス文書の「アバター利用」＝人格を与えて操作し視聴させること、を誰にでも許諾） |
  | commercialUsage | corporation（法人の商用利用まで可） |
  | allowRedistribution / modification | true / allowModificationRedistribution |
  | creditNotation | unnecessary |
  | allowPoliticalOrReligiousUsage | true（allowAntisocialOrHateUsage は false） |

- **Node の入れ方**: 7560 には既に **nvm（`~/.nvm`）で Node v24.22.0 / npm** が入っていたので、そのまま使った（sudo 不要）。新しい機体では次のとおり:
  ```bash
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  nvm install 24
  ```
- **ブラウザ**: 7560 に Chromium は無く、Google Chrome（`/usr/bin/google-chrome`）があったので、それで確認した。ページは Chromium でも同じ
- `server/static/edge/` は git に入れない。`edge/` を変えたら `npm run build` する
- 字幕を音声に合わせて出すのはページ側で行う。フロントの `caption` は契約どおり全文を送る

### 確認方法と計測（B）

ヘッドレス Chrome をエッジにして無人で確認した。

```bash
google-chrome --headless=new --use-fake-ui-for-media-stream --use-fake-device-for-media-stream \
  --use-file-for-fake-audio-capture=<来訪者役 3 発話を 22 秒の無音で区切った WAV> \
  --autoplay-policy=no-user-gesture-required --mute-audio --use-angle=gl-egl \
  --remote-debugging-port=9222 http://localhost:8080/edge/
```

- ページは自動で開始した。`hello`（`caps.aec: true`）を受け、3 往復とも ASR → 応答 → `speak` → ブラウザで再生 → `playback_ended` → `[GATE] unmute` の順になった
- DevTools プロトコルで撮ったスクリーンショットで、次を確認した
  - 胸から上の構図
  - 発話中に口が開く
  - 来訪者・アバターの 2 行字幕
  - 背景が `etajima_autumn` になる
- 計測（`[計測]` の音声開始は、1 文目の `speak` を送った時刻。ブラウザ側のデコード・開始の余裕 0.03 秒が加わる）:

| 往復 | ASR | LLM 1文目確定 | **音声開始** | LLM 完了 |
|---|---|---|---|---|
| 1 挨拶 | 0.29 | 0.63 | **0.97** | 1.38 |
| 2 手続き | 0.40 | 0.60 | **1.07** | 1.85 |
| 3 雑談 | 0.35 | 0.47 | **0.79** | 1.25 |

- 注意: ヘッドレス Chrome の既定はソフトウェア描画（SwiftShader）で、Chrome が CPU を 1400% 使う
  - その状態では VOICEVOX・ASR と CPU を取り合い、音声開始が 1.49〜2.37 秒まで落ちた
  - `--use-angle=gl-egl`（Intel 内蔵 GPU で描画）で上の値になった
  - 画面のある通常の Chrome は GPU で描画するので問題ないはずだが、**キオスク起動ではソフトウェア描画にならないことを確認する**（chrome://gpu）
- ローカル再生モード（`--edge` なし、内蔵スピーカー）の再確認: 0.77 / 1.41 / 0.69 秒
  - 2 往復目は 1 文目が読点なしの長文だったため（1 文目の合成待ち）
  - 同じ質問を 4 回繰り返すと 0.65〜1.11 秒
  - **最終設定での計測 19 回（疑似音声 6・実再生 9・エッジ 3・名前指定の確認 1）のうち、1.3 秒を超えたのはこの 1 回だけ**

### 人間が確認すること（B）

1. **画面のある Chrome で 3 往復**
   - `uv run python -m src.main --mode voice --edge` を起動し、`http://localhost:8080/edge/` を開いて「はじめる」→ マイクを許可
   - 確認すること: 口の動きが声に合っているか、字幕の大きさ、待機・傾聴・考え中・発話の見た目の違い（薄くしたので、物足りなければ `avatar.ts` の `POSES`）
2. **ブラウザのマイク**: Chrome は OS（PipeWire）の既定の入力を使う。内蔵マイクは A-4 の雑音の件があるので、Chrome のサイト設定（アドレスバーの鍵アイコン → マイク）で **Avaya B109** を選ぶ必要がありそう
   - AEC がブラウザ側で効いているか（`hello` の `caps.aec` はログに出る）
3. **VRM のライセンス**: 上の表の転記で、役所のロビーでのデモ用途に問題ないか（アバター利用・法人商用・政治宗教用途とも許諾あり）
4. **キオスク起動**（WP02 の準備）: `--autoplay-policy=no-user-gesture-required` を付けるとボタンなしで始まる。描画がソフトウェアになっていないか

### 次の起動手順

```bash
# サービス（ASR / LLM / VOICEVOX）。VOICEVOX は 8 スレッドで作り直される
docker compose -f deploy/standalone/docker-compose.yaml --env-file deploy/standalone/.env up -d --build
bash scripts/healthcheck.sh

# エッジページのビルド（初回と edge/ を変えたとき）
cd edge && npm ci && npm run build && cd ..

# エッジ（ブラウザ）モード
cd server && uv sync
uv run python -m src.main --mode voice --edge
# → Chrome で http://localhost:8080/edge/ を開き「はじめる」（マイクは Avaya B109 を選ぶ）

# ローカル再生モード（従来）。デバイスは番号より名前の一部で指定する
uv run python -m src.main --mode voice --audio-device Avaya --output-device alsa_output.pci-0000_00_1f.3
```

- `deploy/standalone/.env` には音声デバイスの指定が無い。その場合、ローカル再生は既定デバイスに出て戻らなくなるので、CLI で指定するか、`.env.example` の 2 行（名前での指定）を `.env` に写す

### 品質（B の時点、最終）

- `uv run pytest`: **174 passed**（`-m integration` は 3 passed）
- `uv run mypy --strict src`: Success（18 files）
- `uv run ruff check`: All checks passed
- `edge/`: `npm run typecheck`（tsc --noEmit）と `npm run build` が通る。`npm ci` からのクリーンビルドも確認した
