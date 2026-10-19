# CLAUDE.md — 井戸端AI 開発コンテキスト（Dell Precision 7560）

このファイルは Claude Code が自動で読み込む開発コンテキストです。
コード生成・修正時は必ず本ファイルの制約に従ってください。

## プロジェクト概要

役所のロビーに置く「井戸端AI」。来訪者の話を急かさず聞き、必要なら担当課へ「紹介状」をつなぐ。
映像・音声・会話が建物の外に出ないこと（ローカルAI）が訴求の核。構想は `docs/構想メモ.md`。

- 母体は [NC-prog/townhall-assistant](https://github.com/NC-prog/townhall-assistant) の `server/`（参照用 clone: `~/src/townhall-assistant`、**変更しない**）
- WP00（現在）: standalone プロファイルで「マイク → VAD → Qwen3-ASR → LLM → VOICEVOX → スピーカー」の音声往復のみ。
  アバター・カメラ・エッジ分離・対話FSM は WP01 以降

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
| TTS（VOICEVOX） | `http://127.0.0.1:50021` | 同左 | - | CPU |
| 予備 | - | - | - | 0.20（約3.2GB、YOLO11-pose・多言語TTS用。使い切らない） |

- モデル名は必ず `/models/` プレフィックス付き。API key は `"dummy"`
- Qwen3 の思考モードは既定オフ（サーバに `--reasoning-parser` を付けず、リクエストで `chat_template_kwargs.enable_thinking=false`）
- 拠点（`IDOBATA_SITE=etajima|akiota`）で ASR のコンテキストバイアシング語彙（`server/context/<site>.yaml`）を選ぶ

## 起動・テスト

```bash
# サービング
cp deploy/standalone/.env.example deploy/standalone/.env   # 初回のみ
docker compose -f deploy/standalone/docker-compose.yaml --env-file deploy/standalone/.env up -d --build
bash scripts/healthcheck.sh

# アプリ（server/ で）
cd server
uv sync
uv run python -m src.main --mode text
uv run python -m src.main --mode voice                 # 内蔵マイク
uv run python -m src.main --list-devices
uv run python -m src.main --mode voice --input-wav tests/fixtures/visitor_procedure_ja.wav --output-wav-dir /tmp/out

# 品質
uv run pytest
uv run mypy --strict src
uv run ruff check
```

## コーディング規約

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
scripts/gen_test_audio.py # テスト音声（来訪者役）を VOICEVOX で合成
server/
├── context/<site>.yaml   # 拠点の地名・施設名・課名（ASR コンテキストバイアシング）
├── src/
│   ├── config.py         # 全設定の集約・プロファイル解決
│   ├── audio_input.py    # マイク / WAV 入力（townhall のまま）
│   ├── vad.py            # Silero VAD（townhall のまま）
│   ├── speech_gate.py    # 自己発話ゲート（townhall のまま、後マージン 0.7 秒）
│   ├── asr_client.py     # Qwen3-ASR。言語フィルタなし、検出言語を返す
│   ├── llm_client.py     # vLLM（OpenAI 互換）クライアント（townhall のまま）
│   ├── tts_client.py     # VOICEVOX（townhall のまま）
│   ├── dialogue.py       # 最小対話：固定ペルソナ + 直近 6 ターン（プロセス内のみ）
│   ├── edge_bridge.py    # 旧 unity_bridge.py（WP01 で拡張）
│   ├── caption_feed.py   # 字幕配信（WP01 以降で使用）
│   ├── announcer.py      # テンプレート発話（WP00 では未使用）
│   └── main.py           # text / voice の 2 モード
└── tests/
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
- 会話ログ・音声・映像をディスクに保存する（会話はプロセス内のみ。録音しない。フレームを保存しない）
- `192.168.11.3`（townhall の社内サーバ）をコード・設定・文書に書く
- Live2D 関連（モデル・SDK・Unity プロジェクト）を持ち込む。アバターは three.js + three-vrm（WP01）
- `~/src/townhall-assistant` を変更する
