# WP00 — 7560 ブートストラップ（Claude Code 向け作業指示）

対象機: Dell Precision 7560（RTX A5000 16GB、Ubuntu 26.04.1、nvidia-open 615系、Tailscale名 dell7560、ユーザー fuji）
目的: townhall-assistant の server/ を母体に、井戸端AIの **standalone プロファイル**で「マイク → VAD → Qwen3-ASR → LLM → VOICEVOX → スピーカー」の音声往復を 7560 一台で成立させる。アバター・カメラ・エッジ分離・対話FSMの書き直しは **この WP には含めない**。

## 0. 事前に人間（Norio）が済ませておくこと（sudo を伴うもの）

Claude Code は sudo を使わない。以下が未了なら Claude Code は確認して報告し、その先に進まない。

1. Docker Engine と NVIDIA Container Toolkit が入っていて、`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` が通る。ユーザー fuji が docker グループに入っている。
2. データ用ボリューム `/workspace` は**当面作らない**（2026-10-19 決定。`/home` に約560GB空きがあるため）。将来 SN810 の Windows 領域を縮小して切り出す手順は `docs/7560_workspace_volume.md` に残す。
3. `git config --global user.name / user.email` が設定済み（メールは `NC-prog@users.noreply.github.com`。GitHub のメール非公開設定で push が弾かれるため）。
4. GitHub への push 認証（`gh auth login` 済み、または SSH 鍵）。

## 1. Claude Code への指示（ここから下をそのまま貼る）

---

あなたは Dell Precision 7560（RTX A5000 Laptop 16GB、Ubuntu 26.04.1、ドライバ 615.71.09 / CUDA 13.4）上で作業しています。sudo は使わないでください。sudo が必要な手順に当たったら、実行せずに手順を書き出して止まってください。

**今回の前提（人間が確認済み）**
- Docker 29.1.3 / Compose v2.40.3 / NVIDIA Container Toolkit は導入済みで、`docker run --rm --gpus all ... nvidia-smi` が通る。ユーザー fuji は docker グループに入っている。
- `/workspace` は **ない**。データとモデルはホーム直下に置く：`HF_CACHE_DIR=$HOME/.cache/huggingface`、データは `$HOME/idobata-data`。`/home` は約 560GB 空いている。
- 人間はこのあと離席する。**問い合わせは最小限**にし、以下の方針で自律的に進める：
  - 「1-9. 止まって報告すべきこと」に該当する場合だけ止まる。それ以外の判断（ライブラリの版、ファイル名、テストの細部、代替モデルの選択）は自分で決め、決めた内容を最後の報告にまとめて書く。
  - 止まる場合も、止まるまでにできる作業（コード・設定・テスト・ドキュメント）は全部終えてコミット・push してから止まる。
  - モデルのダウンロード（Qwen3-ASR-1.7B 約4GB、Qwen3-8B-AWQ 約6GB、vLLM イメージ数GB）は時間がかかるので、最初に `docker compose pull` と `huggingface-cli download` をバックグラウンドで開始し、その間にコードの移植を進める。
  - 「1-8. 完了条件」の 3（音声モードで人間が話しかける確認）と 5 のストップウォッチ計測は人間が戻ってから行う。代わりに `server/tests/fixtures/` の WAV（townhall から `test_speech_ja.wav` 等を持ってくる）を `--input-wav` で流す疑似音声モードを実装し、WAV → ASR → LLM → VOICEVOX（音声ファイル出力）まで自動で通ることを確認して完了条件 3 の代替とする。
  - 作業の区切りごとにコミットし、最後に `docs/WP00_result.md` に「やったこと／決めたこと／人間に確認してほしいこと／次に人間がやる手順（音声モードの試し方）」を書いて push する。

### 1-1. リポジトリの準備

- `~/idobata-ai` に `https://github.com/NC-prog/idobata-ai` を clone する（既にあれば `git pull`）。`docs/構想メモ.md` を読み、特に「4. システム構成」「7. 実装の段階 › 機体と実行プロファイル」「8. townhall-assistant からの移植」を把握する。
- `~/src/townhall-assistant` に `https://github.com/NC-prog/townhall-assistant` を clone する（参照用。ここは変更しない。Git LFS の Live2D モデルは不要なので `GIT_LFS_SKIP_SMUDGE=1` で clone してよい）。
- 作業ブランチ `feature/wp00-bootstrap` を `~/idobata-ai` に切る。`git branch --show-current` で確認して報告する。

### 1-2. ディレクトリ構成（この WP で作るもの）

```
idobata-ai/
├── README.md / docs/            （既存）
├── CLAUDE.md                     新規：本機向け開発コンテキスト（下記 1-7）
├── server/                       townhall-assistant/server をコピーして改変
│   ├── pyproject.toml            name を idobata-ai-server に。依存は townhall と同じでよい
│   ├── src/
│   │   ├── config.py             プロファイル対応（1-3）
│   │   ├── audio_input.py        そのまま
│   │   ├── vad.py                そのまま
│   │   ├── speech_gate.py        そのまま
│   │   ├── asr_client.py         言語フィルタを外し、検出言語を返す（1-5）
│   │   ├── tts_client.py         そのまま（VOICEVOX）
│   │   ├── llm_client.py         そのまま
│   │   ├── edge_bridge.py        unity_bridge.py を改名したもの。中身はこの WP では変えない
│   │   ├── caption_feed.py       unity_bridge → edge_bridge の参照名だけ直す
│   │   ├── announcer.py          そのまま（この WP では使わない）
│   │   ├── schemas.py            そのまま
│   │   ├── dialogue.py           新規・最小：1発話 → LLM 1応答（1-6）
│   │   └── main.py               書き直し：text / voice の2モードで往復だけ（1-6）
│   └── tests/                    流用できる unit テスト（vad / speech_gate / asr_client / tts_client / llm_client）を持ってきて通す
├── deploy/
│   └── standalone/
│       ├── docker-compose.yaml   ASR + LLM + VOICEVOX（1-4）
│       └── .env.example
└── scripts/
    └── healthcheck.sh            townhall の server/scripts/healthcheck.sh を元に、ポートと名前を合わせる
```

持ってこないもの: `unity/`、`server/agenda/`、`server/outputs/`、`session_manager.py`、`summarizer.py`、`web_panel.py`、`static/panel.html`、それらのテスト。これらは WP01 以降で対話FSMとして書き直すので、今は入れない。

### 1-3. プロファイル（config.py）

`IDOBATA_PROFILE` 環境変数（`.env` で与える）で `standalone` / `site` を切り替える。この WP で実装するのは standalone のみだが、config の構造は両方を見据える。

- `standalone`: ASR・LLM・TTS すべて `http://127.0.0.1` の各ポート。
- `site`: ASR・TTS は `127.0.0.1`、LLM は `IDOBATA_BACKEND_URL`（Tailscale 経由の T2／3660）を指す。未設定なら起動時にエラーで止める。
- モデル名は townhall と同じく `/models/asr`、`/models/llm` の `/models/` プレフィックス。API key は `"dummy"`。
- townhall の config にある `192.168.11.3` 固定のエンドポイントは**すべて消す**。残っていたら grep で潰す。

### 1-4. サービング（deploy/standalone/docker-compose.yaml）

A5000 16GB の配分。`gpu-memory-utilization` は「GPU全体に対する割合」なので 16GB 基準で決める。

| サービス | イメージ／モデル | ポート | GPU割当 | 備考 |
|---|---|---|---|---|
| asr | townhall の `deploy/asr/Dockerfile`（vLLM v0.27.1）を流用、`Qwen/Qwen3-ASR-1.7B` | 8005 | `--gpu-memory-utilization 0.30`（約4.8GB）、`--max-model-len 4096`、`--enforce-eager` | served-model-name `/models/asr` |
| llm | 同じ vLLM イメージ、`Qwen/Qwen3-8B-AWQ`（Qwen 公式リポジトリ）。もし取得できなければコミュニティ版は使わず、`Qwen/Qwen3-4B` の bf16（約8GB）で代替して報告する。FP8 は A5000（Ampere）では使えない | 8001 | `--gpu-memory-utilization 0.45`（約7.2GB）、`--max-model-len 8192` | served-model-name `/models/llm`。思考モードは既定でオフ（`--reasoning-parser` は付けず、リクエスト側で `enable_thinking: false` を渡す） |
| voicevox | `voicevox/voicevox_engine:cpu-latest` | 50021 | なし（CPU） | townhall の OPERATION.md と同じ |

- HF キャッシュは `${HF_CACHE_DIR:-${HOME}/.cache/huggingface}` をマウント。将来 `/workspace` を切り出したら `.env` の 1 行で切り替えられるようにしておく。
- 残り約4GB（0.25）は WP01 以降の YOLO11-pose と将来の多言語TTS のために空けておく。この WP で使い切らないこと。
- 両 vLLM コンテナを同時に上げ、`nvidia-smi` で実使用量を記録して報告する（配分の見直し材料にする）。
- `restart: unless-stopped`。healthcheck は townhall と同じ `curl /v1/models`。

### 1-5. asr_client.py の変更点

- townhall は検出言語が `ja` 以外のセグメントを破棄している。これを**外す**。`TranscriptChunk` に `language: str` を持たせ、呼び出し側へ返す。
- 静的コンテキスト（コンテキストバイアシング）の中身は、アジェンダ用語の代わりに `config` から読む「地名・施設名・課名のリスト」にする。この WP では江田島・安芸太田の市役所名・町役場名・主要施設名を各10語程度、仮の YAML（`server/context/etajima.yaml`、`akiota.yaml`）に置き、`IDOBATA_SITE` 環境変数で選ぶ。

### 1-6. main.py と dialogue.py（最小）

- `--mode text`: 標準入力の1行 → `dialogue.respond()` → 応答テキストを表示し、VOICEVOX で再生（出力デバイス指定可）。
- `--mode voice`: マイク → SpeechGate → VAD → ASR → `dialogue.respond()` → VOICEVOX 再生。townhall の音声パイプラインをそのまま使う。SpeechGate で自己発話の回り込みを防ぐ（townhall と同じ後マージン 0.7 秒）。
- `dialogue.respond(text, language)` は、固定のシステムプロンプト（井戸端AIの最小ペルソナ：「市役所のロビーにいる、話を急かさない聞き役。敬語だが硬くない。一度に長く話さない（2〜3文）。手続きの詳細を断定しない」）で LLM に投げ、直近 6 ターンだけを履歴として持つ。履歴はプロセス内のみで、ディスクに書かない。
- 会話ログの保存は**しない**（構想メモ 5. の方針）。デバッグ用のログは `logging` で標準出力のみ、個人を特定しうる本文は DEBUG レベルに落とす。

### 1-7. CLAUDE.md（新規）

townhall の CLAUDE.md を下敷きに、本機向けに書き換える。必ず含めること:

- 実行環境（7560、A5000 16GB、Ubuntu 26.04.1、nvidia-open 615、Python は uv で 3.12 を固定。システム Python 3.14 は使わない）
- プロファイルとエンドポイントの表（standalone / site）
- 「やってはいけないこと」: Ollama／Whisper 不使用、pip 直叩き禁止、sudo 禁止、会話ログ・音声・映像をディスクに保存しない、`192.168.11.3` を書かない、Live2D 関連を持ち込まない
- コーディング規約（型ヒント・mypy strict・config 集約・外部呼び出しのタイムアウトとリトライ・print 禁止）は townhall のまま
- ブランチ運用（main 直コミット禁止、feature/*）と完了の定義（コミットハッシュ・push 済み・`git status` クリーン・テスト結果の明記）も townhall のまま

### 1-8. 完了条件

1. `docker compose -f deploy/standalone/docker-compose.yaml up -d` で 3 サービスが healthy。`scripts/healthcheck.sh` が全 OK。
2. `uv run python -m src.main --mode text` で、入力 → 応答 → VOICEVOX 再生が通る。
3. `uv run python -m src.main --mode voice` で、内蔵マイクに話しかけて応答が返り、自己発話の回り込みが起きない（3往復以上）。
4. `uv run pytest` / `uv run mypy --strict src` / `uv run ruff check` が通る。
5. `nvidia-smi` の実使用量（ASR / LLM それぞれ）と、発話終了→応答音声開始までの体感レイテンシ（ストップウォッチで3回）を `docs/WP00_result.md` に記録する。
6. `feature/wp00-bootstrap` に push し、コミットハッシュを報告する。main へのマージは人間が判断する。

### 1-9. 止まって報告すべきこと

- Docker / NVIDIA Container Toolkit が使えない。
- `Qwen/Qwen3-8B-AWQ` 相当が見つからず代替した。
- vLLM v0.27.1 で Qwen3-ASR か Qwen3-8B が起動しない（イメージの版上げが必要なら、候補の版と理由を書いて止まる）。
- VRAM が足りず配分を変えた。
- 内蔵マイク／スピーカーが PipeWire で取れない（デバイス一覧を貼って止まる）。

---

## 2. この WP の次（参考、着手しない）

- WP01: エッジ分離。`edge_bridge` に音声上り・TTS音声下りを追加し、7560 上の Chromium（three.js + three-vrm の最小ページ）をエッジ役にする。
- WP02: `camera_client`（YOLO11-pose）と三段階声かけ。
- WP03: 対話FSM（session_manager の書き直し）と紹介状。
