# WP01 — 応答の速さと、エッジ分離＋ブラウザアバター（Claude Code 向け作業指示）

対象機: 7560（standalone、WP00 完了、WP00.5 で TTS は VOICEVOX 継続と決定）
前提の決定（2026-10-21）:
- 日本語 TTS は **VOICEVOX で確定**（高齢者に明瞭。Qwen3-TTS は日本語の抑揚が悪く、7560 では RTF≈1.0 で遅い）。**英語は Qwen3-TTS の方が良好**だったので、多言語（英・中）は site プロファイル（GPU に余裕のある機体）で Qwen3-TTS を使う方針。今回は扱わない。
- 応答の長さは今のまま（短くしない）。
- LLM の GPU 割当は 0.45 のまま。予備 3.2GB は WP02 の YOLO11-pose 用に残す。

WP01 は **A → B の 2 段階**。A を終えて push してから B に入る。A だけで止まっても価値がある。

---

## A. 応答の速さ（同一機のまま。エッジ分離なし）

目標: 発話終了→音声開始を **1.0 秒前後**（現状 1.95〜2.89 秒）。固有名詞の読み間違いをなくす。

### A-1. 文単位の先行合成（tts_client / main）

- LLM の応答を **ストリーミング**で受け（`stream=True`）、句点「。」「？」「！」と改行で文に切る。1 文確定するごとに VOICEVOX へ合成を投げる（合成は直列で可。VOICEVOX は CPU で RTF 0.13 なので十分）。
- 再生キューを作り、1 文目の WAV ができた時点で再生を始め、以降の文は合成でき次第キューに積む。文と文の間に無音を挟まない（VOICEVOX の `postPhonemeLength` を 0.1 秒程度にし、間はそれで作る）。
- SpeechGate は「最初の文の再生開始で mute、キューが空になって最後の再生が終わった時点で schedule_unmute」。途中で開けない。
- `[計測]` に `音声開始` のほか `LLM 1文目確定` を追加。
- `dialogue.respond()` は非同期ジェネレータで文を yield する形に変える。履歴には最終的な全文を 1 ターンとして積む。

### A-2. 固有名詞の読み（VOICEVOX ユーザー辞書）

- `server/context/<site>.yaml` の各語に `reading`（カタカナ）と任意の `accent`（アクセント核位置、省略時は 1）を持てるようにし、起動時に VOICEVOX の `POST /user_dict_word` で登録する（既に同じ surface があれば `PUT` で更新。`GET /user_dict` で突き合わせ）。
- 最低限、江田島は「江田島（エタジマ）」「能美（ノウミ）」「大柿（オオガキ）」「沖美（オキミ）」「小用（コヨウ）」「切串（キリクシ）」「中町（ナカマチ）」、安芸太田は「安芸太田（アキオオタ）」「加計（カケ）」「戸河内（トゴウチ）」「筒賀（ツツガ）」「三段峡（サンダンキョウ）」「温井（ヌクイ）」を入れる。課名（市民課・税務課・福祉課など）は一般語なので登録不要。**読みが確信できない語は登録せず、`docs/WP01_result.md` に「人間が確認」として列挙する**。
- 同じ語彙は ASR のコンテキストバイアシングにも既に使っているので、YAML は 1 つのまま。

### A-3. 完了条件（A）

- `--mode voice --audio-device 0 --output-device 20` で 3 往復し、`[計測]` の `音声開始` が 3 回とも 1.3 秒未満。
- `--mode text` で「能美の秋祭りはいつですか」と入れ、応答音声で「のうみ」と読まれる（人間が耳で確認。Claude Code は合成 WAV を ASR に掛けて「能美」と戻ることを代替確認にする）。
- pytest / mypy strict / ruff 通過。`feature/wp01-edge-avatar` に push、コミットハッシュ報告。

---

## B. エッジ分離＋ブラウザアバター（同一機で、ブラウザをエッジ役に）

目標: 7560 の Chromium を「エッジ」にして、マイク・スピーカー・アバター描画をブラウザ側に移す。フロントAI（Python）はブラウザから音声を受け、音声と口パクを返す。**7560 一台で standalone デモが成立する**状態にする。別機のエッジ PC への配置は WP02。

### B-1. エッジ⇄フロントの契約（edge_bridge）

`edge_bridge.py`（旧 unity_bridge）を、**1 本の WebSocket**で JSON（テキストフレーム）と音声（バイナリフレーム）を混在させる方式に拡張する。ポートは 8765 のまま。静的ページは同じ FastAPI プロセスから `http://localhost:8080/edge/` で配信する（WP00 の web_panel は持ってきていないので、`edge_server.py` として小さく新設）。

上り（エッジ → フロント）
| 種別 | 内容 |
|---|---|
| JSON `hello` | `{type:"hello", edge_id, sample_rate:16000, caps:{aec:true/false}}` |
| バイナリ | PCM16 mono 16kHz、20ms（320 サンプル）ごと。ブラウザの AudioWorklet で取り、ダウンサンプルして送る |
| JSON `playback_ended` | `{type:"playback_ended", utterance_id}` 再生キューが空になった通知（SpeechGate の unmute に使う） |
| JSON `interrupt` | 予約（WP03 の割り込み用。今回は送らない） |

下り（フロント → エッジ）
| 種別 | 内容 |
|---|---|
| JSON `state` | `{type:"state", value:"idle"|"listening"|"thinking"|"speaking"}` アバターの待機・傾聴・考え中・発話の切替 |
| JSON `caption` | `{type:"caption", role:"user"|"assistant", text, final}` 字幕（既存の caption を流用） |
| JSON `speak` | `{type:"speak", utterance_id, seq, text, mouth:[{t, v, d}...]}` 1 文ぶんの口パク時系列（後述）。直後にその文の音声バイナリが続く |
| バイナリ | WAV（VOICEVOX 出力、24kHz mono）を 1 文ずつ。ブラウザは `decodeAudioData` して再生キューに積む |
| JSON `theme` | `{type:"theme", site:"etajima", season:"autumn"}` 衣装・背景の切替（今回は背景画像の差し替えだけ） |

- 口パク `mouth` は VOICEVOX の `audio_query` の `accent_phrases[].moras[]` から作る：各モーラの `consonant_length + vowel_length` を累積して開始時刻 `t`、母音 `vowel`（a/i/u/e/o/N/cl）を VRM の口形 `aa/ih/ou/ee/oh`（N・cl は閉口）に写し、`d` に長さ。`prePhonemeLength` を先頭に加える。これで**音素に同期した口パク**になる（音量包絡より良い）。
- フロント側の `audio_input.py` に、ローカルマイクの代わりに WebSocket から PCM を受ける `EdgeAudioInput` を追加。VAD 以降のパイプラインは変更しない。
- `tts_client.play()` のローカル再生は残す（`--edge` を付けない時は従来どおり）。`--edge` 指定時は再生の代わりに `speak`＋バイナリをエッジへ送り、`playback_ended` を待って gate を開く。
- ブラウザは `getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true}})` で取る。**ブラウザ側に AEC があるのが、ALSA 直取りに対しての大きな利点**。`hello.caps.aec` を true で送る。SpeechGate は当面そのまま（AEC を信じて緩めるのは WP03）。

### B-2. エッジページ（`edge/` ディレクトリ、TypeScript + Vite）

- `edge/` に Vite + TypeScript のプロジェクトを作る（npm、Node は 7560 に入っていなければ `uv tool` ではなく nvm 等 sudo 不要の方法で入れ、手順を記録）。ビルド成果物を `server/static/edge/` に出し、FastAPI から配信。
- three.js + `@pixiv/three-vrm` で VRM を 1 体表示。カメラは胸から上、背景は `theme` で差し替える画像（`edge/public/backgrounds/<site>_<season>.jpg`。今回はプレースホルダとして単色グラデーションか、権利の明確なフリー素材 1 枚。実写は後で市から受け取る）。
- VRM モデルは**ライセンスを確認したサンプル**を使う（three-vrm リポジトリの examples に同梱のモデル、または VRoid の公式サンプル）。採用したモデル名・配布元・ライセンス条項を `docs/WP01_result.md` に転記する。**推測で「使ってよい」と判断しない**。本番用アバターは人間が VRoid Studio で作る。
- 状態ごとの見た目：idle は緩い呼吸とまばたき、listening は少し前傾して相手を見る（視線は正面固定でよい）、thinking は視線を少し外す、speaking は口パク＋軽い頷き。表情は VRM 1.0 のプリセット（happy/relaxed 等）を状態に合わせて薄く乗せる程度。
- 字幕は画面下に、来訪者の発話（確定のみ）とアバターの発話を 2 行で。フォントは大きめ（高齢者）。
- 起動は `http://localhost:8080/edge/` を Chromium で開くだけ。初回はマイク許可のダイアログが出る。

### B-3. 完了条件（B）

- `--mode voice --edge` でフロントを起動し、Chromium で `/edge/` を開いて 3 往復。ブラウザのマイクで聞き取り、ブラウザのスピーカーで応答、口が音素に合わせて動き、`playback_ended` の後に gate が開く。
- `[計測]` の `音声開始` が A と同等（1.3 秒未満）。
- ローカル再生モード（`--edge` なし）も壊れていない（A の確認を再実行）。
- pytest / mypy strict / ruff、`edge/` は `npm run build` と `tsc --noEmit` が通る。
- `docs/WP01_result.md`：やったこと／決めたこと（VRM の出所とライセンス、Node の入れ方）／人間に確認してほしいこと／次の起動手順。CLAUDE.md の構成・起動例を更新。

---

## 止まって報告すべきこと

- VOICEVOX の `/user_dict_word` が使えない（エンジン版の問題）。
- ライセンスが明確な VRM サンプルが見つからない（その場合、VRM なしで「丸い顔のプレースホルダ」を three.js の形状で描いて先に進み、報告する）。
- Node を sudo なしで入れられない。
- ブラウザの `getUserMedia` が localhost で拒否される（Chromium の版・フラグを記録して止まる）。

## 着手しないこと（次以降）

- WP02: 別機のエッジ PC への配置（plain HTTP での `getUserMedia` にはキオスク起動時の `--unsafely-treat-insecure-origin-as-secure=http://<front>:8080` か自己署名証明書が要る。その判断は WP02）、YOLO11-pose の三段階声かけ、季節テーマの実データ。
- WP03: 対話 FSM（傾聴／紹介状／同意）、割り込み（バージイン）、職員パネル。
