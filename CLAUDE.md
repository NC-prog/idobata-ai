# CLAUDE.md — 井戸端AI 開発コンテキスト（Dell Precision 7560）

このファイルは Claude Code が自動で読み込む開発コンテキストです。
コード生成・修正時は必ず本ファイルの制約に従ってください。

## プロジェクト概要

役所のロビーに置く「井戸端AI」。来訪者の話を急かさず聞き、必要なら担当課へ「紹介状」をつなぐ。
映像・音声・会話が建物の外に出ないこと（ローカルAI）が訴求の核。構想は `docs/構想メモ.md`。

- 母体は [NC-prog/townhall-assistant](https://github.com/NC-prog/townhall-assistant) の `server/`（参照用 clone: `~/src/townhall-assistant`、**変更しない**）
- WP00: standalone プロファイルで「マイク → VAD → Qwen3-ASR → LLM → VOICEVOX → スピーカー」の音声往復
- WP01（現在）: 応答の文単位先行合成（音声開始 約 1 秒）、固有名詞の VOICEVOX ユーザー辞書、
  エッジ分離（ブラウザがマイク・スピーカー・VRM アバターを持つ、`--edge`）。結果は `docs/WP01_result.md`
- WP03（現在）: ペルソナ（拠点の固定事実をプロンプトに直書き）と知識パック（自治体サイトのクロール →
  Ruri v3 埋め込み → LanceDB、気象庁の予報）。手続きの詳細は出典があるときだけ言う。結果は `docs/WP03_result.md`
- 日本語 TTS は VOICEVOX で確定（WP00.5）。カメラ・YOLO11-pose・別機のエッジ PC は WP02、紹介状・職員パネルは WP04

## 実行環境（重要）

| 項目 | 値 |
|------|-----|
| 機体 | Dell Precision 7560（開発機＝持ち回りデモ機） |
| GPU | NVIDIA RTX A5000 Laptop 16GB（Ampere。**FP8 不可**） |
| OS | Ubuntu 26.04.1（ネイティブ。WSL2 ではない） |
| ドライバ | nvidia-open 615 系（615.71.09 / CUDA 13.4） |
| Python | **uv で 3.12 を固定**（`server/.python-version`）。システムの Python 3.14 は使わない |
| サービング | Docker Compose（`deploy/standalone/docker-compose.yaml`）、vLLM v0.27.1 |
| データ | `/workspace` は無い。HF キャッシュ `~/.cache/huggingface`、データ `~/idobata-data`（`.env` の 1 行で切替可） |

## プロファイルとエンドポイント

`IDOBATA_PROFILE`（`deploy/standalone/.env`）で切り替える。設定は `server/src/config.py` に集約。

| サービス | standalone | site | モデル名（API 指定） | GPU 割当（standalone） |
|---|---|---|---|---|
| ASR（Qwen3-ASR-1.7B） | `http://127.0.0.1:8005/v1` | 同左 | `/models/asr` | 0.35（約5.6GB、実測 5.2GB） |
| LLM（Qwen3-8B-AWQ） | `http://127.0.0.1:8001/v1` | `IDOBATA_BACKEND_URL`（Tailscale 経由の T2／3660。未設定なら起動時エラー） | `/models/llm` | 0.45（約7.2GB、実測 7.3GB、`--enforce-eager`） |
| TTS（VOICEVOX） | `http://127.0.0.1:50021` | 同左 | - | CPU（8 スレッド、`VOICEVOX_CPU_THREADS`） |
| エッジ（`--edge`） | ページ `http://localhost:8080/edge/`、WebSocket `ws://127.0.0.1:8765` | 同左 | - | - |
| 予備 | - | - | - | 0.20（約3.2GB、YOLO11-pose・多言語TTS用。使い切らない） |

- モデル名は必ず `/models/` プレフィックス付き。API key は `"dummy"`
- Qwen3 の思考モードは既定オフ（サーバに `--reasoning-parser` を付けず、リクエストで `chat_template_kwargs.enable_thinking=false`）
- 拠点（`IDOBATA_SITE=etajima|akiota`）で `server/context/<site>.yaml` を選ぶ。語彙（ASR バイアシング・VOICEVOX 辞書）と
  `facility`（施設名・開庁時間・課と担当と階・緊急連絡先。**市町の公式サイトで確認したものだけ**）を持つ
- 知識パックは `~/idobata-data/kb/<site>/`（`kb/build.py` で作る）
- ネットワーク: **窓口機から許可ドメインへの取得は可。会話・音声・映像は送らない**（2026-10-21 改定）。
  外に出るのは知識パックの取得と、「調べる」（WP04）の検索語だけ。許可ドメインは `kb/sources/<site>.yaml`。
  一般のウェブ検索（`IDOBATA_LOOKUP_WEB_SEARCH`）は既定オフ
- 同意付きの行為（WP04）: LLM が応答の末尾に `<propose:lookup|refer|share>` を付け、決まった文で提案し、
  返事は規則で判定する（`server/context/actions.yaml`）。相槌は `server/context/backchannel.yaml`、
  紹介状のタグは統制語彙 `server/context/tags.yaml`。紹介状の担当課は規則だけで決める（`server/context/procedures.yaml` の
  別名表で発話と `facility.departments[].handles` を突き合わせ、一意でなければ「未特定」で保存して職員が職員パネルで付ける。
  来訪者に課を聞き返さない・LLM に課を推測させない）。procedures.yaml は ASR の語彙（先頭に置く）・VOICEVOX 辞書・
  聞き取り間違いの対応表（asr_fixes。職員パネルから足した行は `~/idobata-data/asr_fixes.jsonl`）にも使う。
  紹介状の同意の後は、手続きごとの聞き取り（procedures.yaml の intake、最大 3 問）→ お名前・地区・連絡方法を伺う。
  **氏名・地区・電話番号は字幕・エッジ画面・コンソール・ログに出さない**（「お名前：伺いました」と伏せる。保存は紹介状 JSON だけ）
- 案内役の人物像は `server/context/persona.yaml`（WP05。肩書 `role_title` は `IDOBATA_ROLE_TITLE` で差し替え可）。
  人物像は【話し方の決まり】の後ろに置き、事実・出典・設備と課の安全装置・日本語のみの規則が優先。
  人物像と食い違う言い方（contradictions）は読み上げない。資料の無い地区の様子は作らず教えてもらう
- 短い返事（うん・はい・そう）: 返事待ちの間だけ VAD のしきい値と最小発話長を下げ、短い区間は無音を足して ASR に渡す。
  日本語の会話で短い区間が他言語と判定されたら日本語に固定して聞き直す（`VADConfig` / `ASRConfig`）
- 挨拶は時間帯で変える（`src/timeofday.py`。5〜10 時 おはようございます、10〜17 時 こんにちは、17 時以降 こんばんは）。
  応答・相槌・定型文（`{greeting}`）・アナウンスで同じ判定を使い、LLM が別の時間帯の挨拶を言ったら読み上げ前に直すASR は返事待ちの間だけ言語を固定する
  （`deploy/standalone/asr/chat_template.jinja` の `force_language`）

## 起動・テスト

```bash
# サービング
cp deploy/standalone/.env.example deploy/standalone/.env   # 初回のみ
docker compose -f deploy/standalone/docker-compose.yaml --env-file deploy/standalone/.env up -d --build
bash scripts/healthcheck.sh

# アプリ（server/ で）
cd server
uv sync
uv run python -m src.main --mode text --output-device alsa_output.pci-0000_00_1f.3
uv run python -m src.main --mode voice --audio-device hw:0,0 --output-device alsa_output.pci-0000_00_1f.3
uv run python -m src.main --list-devices   # デバイスは番号か名前の一部。番号は USB 機器の抜き差しでずれる
uv run python -m src.main --mode voice --input-wav tests/fixtures/visitor_procedure_ja.wav --output-wav-dir /tmp/out

# 知識パック（kb/ は別の uv 環境。埋め込みは CPU）。--weather-only で天気だけ
CUDA_VISIBLE_DEVICES= uv run --project ../kb python ../kb/build.py --site etajima
CUDA_VISIBLE_DEVICES= uv run --project ../kb python ../kb/build.py --site etajima --daily   # 更新の速い区画だけ（差分）
bash ../deploy/standalone/install-kb-timers.sh   # 自動更新（毎朝 6:00 差分・毎週日曜 3:00 全体・毎時 05 分天気。有効化済み）
uv run python -m tests.eval.run --site etajima   # 評価セット 30 問（WP03 の 20 問＋人物像 10 問。LLM と知識パックを使う）
uv run python -m tests.eval.run --site etajima --set smalltalk   # 雑談 10 問（締めの定型で終わる割合を数える）

# エッジ（ブラウザ）モード: マイク・スピーカー・アバターはブラウザ側
(cd ../edge && npm ci && npm run build)    # 初回と edge/ を変えたとき。成果物は server/static/edge/
uv run python -m src.main --mode voice --edge   # → Chrome で http://localhost:8080/edge/ を開き「はじめる」

# WP04: 職員パネル・行為（.env で設定。職員パネルは IDOBATA_STAFF_PASSWORD を入れたときだけ有効）
#   IDOBATA_AVATAR_NAME=いどちゃん          アバターの呼び名（仮）
#   IDOBATA_STAFF_PASSWORD=...              /staff/ の Basic 認証（ユーザー staff）。LAN からだけ開ける
#   IDOBATA_EDGE_HTTP_HOST=0.0.0.0          職員の PC から /staff/ を開くとき（既定は 127.0.0.1）
#   IDOBATA_REFERRAL_RETENTION_DAYS=30      紹介状の保存日数（起動時に古いものを消す。0 で前日以前を消す）
#   IDOBATA_LOOKUP_WEB_SEARCH=0             一般のウェブ検索（既定オフ。評価では使わない）
uv run python -m src.main --mode voice --edge   # → 職員は http://<窓口機>:8080/staff/ （10 秒ごとに更新）

# 品質
uv run pytest
uv run pytest -m integration   # 実サービス（ASR / LLM / VOICEVOX）に繋ぐもの
uv run mypy --strict src
uv run ruff check
(cd ../edge && npm run typecheck && npm run build)
```

- 7560 では既定の出力デバイス（PipeWire の ALSA プラグイン）への再生が戻らない。ローカル再生では必ず `--output-device` を指定する
- Node は nvm で入っている（`~/.nvm`、v24）。sudo 不要

## コーディング規約

- **報告・コミットメッセージ・結果文書（docs/WP*_result.md）は日本語で書く**
- 型ヒント必須（mypy strict が通ること）
- docstring・コメントは日本語でよい
- 設定は `server/src/config.py` に集約（dataclass）。マジックナンバー散在禁止
- 外部サービス呼び出しは必ずタイムアウトとリトライを実装
- ログは `logging` モジュール。print 禁止（CLI のユーザー向け出力を除く）
- 個人を特定しうる本文（発話・応答テキスト）は DEBUG レベルでのみログに出す
- テストは pytest。`uv run pytest` で実行（実サービスに繋ぐものは `@pytest.mark.integration`）

## ディレクトリ構成

```
CLAUDE.md
deploy/standalone/        # docker-compose.yaml（ASR + LLM + VOICEVOX）、.env.example、vllm/Dockerfile
scripts/healthcheck.sh    # 前提サービスの一括ヘルスチェック
scripts/make_fixtures.py  # テスト音声（来訪者役）を VOICEVOX で合成
server/
├── context/<site>.yaml   # 拠点の地名・施設名・課名（ASR コンテキストバイアシング）
├── src/
│   ├── config.py         # 全設定の集約・プロファイル解決
│   ├── audio_input.py    # マイク / WAV 入力（townhall のまま）
│   ├── vad.py            # Silero VAD（townhall のまま）
│   ├── speech_gate.py    # 自己発話ゲート（townhall のまま、後マージン 0.7 秒）
│   ├── asr_client.py     # Qwen3-ASR。言語フィルタなし、検出言語を返す
│   ├── site_context.py   # 拠点 YAML（語・読み・アクセント）。ASR バイアシングと VOICEVOX ユーザー辞書の両方に使う
│   ├── llm_client.py     # vLLM（OpenAI 互換）クライアント。chat_stream でストリーミング
│   ├── tts_client.py     # VOICEVOX。ユーザー辞書の同期、StreamPlayer（文ごとの WAV を切れ目なく再生）
│   ├── dialogue.py       # 最小対話：固定ペルソナ + 直近 6 ターン（プロセス内のみ）。応答を文ごとに yield
│   ├── persona.py        # システムプロンプト（固定事実 + 参考資料 + 今日の日付・天気）
│   ├── knowledge.py      # 知識パックの検索（Ruri v3 + 文字 2-gram BM25 のハイブリッド）と天気
│   ├── speech_output.py  # 応答音声の出力先（LocalSpeaker / WavDirWriter / EdgeSpeaker）
│   ├── lipsync.py        # audio_query → 口形の時系列（音素同期の口パク）
│   ├── edge_bridge.py    # エッジとの WebSocket（JSON + PCM/WAV バイナリ、8765）
│   ├── edge_server.py    # エッジページの静的配信（FastAPI、8080 /edge/）と職員パネル（/staff/、Basic 認証・LAN のみ）
│   ├── backchannel.py    # 相槌（事前合成、発話の種類で選ぶ）
│   ├── actions.py        # 同意付きの行為（提案 → 返事 → 実行）と来訪記録
│   ├── lookup.py         # 調べる（許可ドメインだけ・5 秒で打ち切り・見つからなければ夜のビルドに回す）
│   ├── referrals.py      # 紹介状（受付番号・保存・期限・安芸太田は CSV）、来訪記録、小ネタ、今日の集計
│   ├── procedures.py     # 手続きの常用語（procedures.yaml）と担当課の規則による決定
│   ├── caption_feed.py   # 字幕配信（WP03 で使用予定）
│   ├── announcer.py      # テンプレート発話（未使用）
│   └── main.py           # text / voice の 2 モード、--edge
├── static/edge/          # edge/ のビルド成果物（git 管理外）
├── static/staff.html     # 職員パネル
└── tests/
kb/                       # 知識パックのビルド（sources/<site>.yaml、build.py、README.md。pack/ は git 管理外）
edge/                     # エッジページ（Vite + TypeScript、three.js + @pixiv/three-vrm）
├── src/                  # main.ts / avatar.ts / audio.ts / link.ts / protocol.ts
└── public/               # mic-processor.js（AudioWorklet）、models/（VRM サンプル）、backgrounds/
```

## ブランチ運用

- **main ブランチ**には直接コミットしない
- 機能開発は `feature/*` ブランチで行う
- 作業開始時に `git branch --show-current` でブランチを確認・報告すること
- main へのマージは E2E 確認後に人間が判断する

## タスク完了の定義

- **実装完了の報告時はコミットハッシュと push 済みを必ず明記する。
  未コミットの実装を完了と報告しない**
- 作業ツリーに実装が残ったままの状態は「完了」ではない。
  誤操作で消失し得るため、E2E 確認はコミット済みの状態に対して行う
- 報告に含める項目: コミットハッシュ / push 先ブランチ / `git status` がクリーンであること
- テスト・型・Lint の結果（`uv run pytest` / `mypy --strict` / `ruff`）も併せて報告する

## やってはいけないこと

- Ollama（localhost:11434）を参照する、Whisper / faster-whisper を依存に追加する
- `pip install` を直接実行する（uv を使う。`uv add` / `uv run`）
- `sudo` を使う（必要な手順に当たったら、実行せず手順を書き出して止まる）
- 会話ログ・音声・映像をディスクに保存する（会話はプロセス内のみ。録音しない。フレームを保存しない）。
  **例外（WP04）: 本人が同意した紹介状（`~/idobata-data/referrals/`、既定 30 日）と匿名の小ネタ
  （`tidbits/`、7 日）、本文を含まない匿名の来訪記録（`visits/`、1 年）だけは保存してよい**。
  保持期間は設定で持ち、起動時に期限切れを消す
- `192.168.11.3`（townhall の社内サーバ）をコード・設定・文書に書く
- Live2D 関連（モデル・SDK・Unity プロジェクト）を持ち込む。アバターは three.js + three-vrm（WP01）
- `~/src/townhall-assistant` を変更する
