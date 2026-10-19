# WP00 結果報告 — 7560 ブートストラップ

実施: 2026-10-19（Claude Code 自律実行、人間は離席中）
ブランチ: `feature/wp00-bootstrap`（main へのマージは人間判断）
機体: Dell Precision 7560 / RTX A5000 Laptop 16GB / Ubuntu 26.04.1 / ドライバ 615.71.09（CUDA 13.4）

## 0. 結論（先に読む）

- **3 サービス（ASR・LLM・VOICEVOX）は 7560 一台で同時に healthy。**`scripts/healthcheck.sh` は 3/3 OK。
- **WAV → ASR → LLM → VOICEVOX（音声ファイル）の往復を 3 往復、無人で確認済み**（完了条件 3 の代替）。
- **止まって報告すべき事項が 2 件**（1-9 該当）。詳細は「3. 人間に確認してほしいこと」:
  1. **VRAM が足りず配分を変えた**：ASR を 0.30 → **0.35** に上げた（0.30 では Qwen3-ASR-1.7B が起動しない）。残りの予備は 0.25 → 0.20。
  2. **Python からスピーカー・マイクを開けない**：PipeWire 自体はマイク・スピーカーとも認識しているが、`sounddevice` が使う **`libportaudio2` が未インストール**（入れるには sudo が要る）。このため完了条件 2 の「VOICEVOX 再生」と完了条件 3（実マイク）はまだ確認できていない。
- 代替モデルは使っていない。`Qwen/Qwen3-8B-AWQ`（Qwen 公式）を取得・起動できた。vLLM v0.27.1 のまま、版上げは不要だった。

## 1. やったこと

| # | 内容 | 状態 |
|---|---|---|
| 1-1 | `~/idobata-ai` で `feature/wp00-bootstrap` を作成（`git branch --show-current` → `feature/wp00-bootstrap`）。`~/src/townhall-assistant` を `GIT_LFS_SKIP_SMUDGE=1` で clone（参照のみ、変更なし） | 済 |
| 1-2 | `server/` を townhall から移植（指示どおりの取捨）、`deploy/standalone/`、`scripts/healthcheck.sh`、`CLAUDE.md` | 済 |
| 1-3 | `config.py`：`IDOBATA_PROFILE`（standalone / site）。site で `IDOBATA_BACKEND_URL` が無ければ起動時に `ConfigError` で止まる。`192.168.11.3` はリポジトリから消した（テストで検査） | 済 |
| 1-4 | `docker-compose.yaml`：asr(8005) / llm(8001) / voicevox(50021)、すべて `127.0.0.1` だけで待ち受け。HF キャッシュは `${HF_CACHE_DIR:-${HOME}/.cache/huggingface}` | 済（配分変更あり） |
| 1-5 | `asr_client.py`：言語フィルタを外した。`TranscriptChunk.language`（ISO 639-1）を返す。静的コンテキストは `server/context/{etajima,akiota}.yaml`（`IDOBATA_SITE` で選ぶ） | 済 |
| 1-6 | `dialogue.py`（固定ペルソナ＋直近 6 ターン、プロセス内のみ）、`main.py`（text / voice / 疑似音声） | 済 |
| 1-7 | `CLAUDE.md`（実行環境、プロファイル表、禁止事項、規約、ブランチ運用、完了の定義） | 済 |

### 完了条件の状況

| # | 条件 | 結果 |
|---|---|---|
| 1 | `docker compose ... up -d` で 3 サービス healthy、`healthcheck.sh` 全 OK | **OK**。`down` から `up -d` 一発で全 healthy まで 156 秒（`up -d` は llm が asr の healthy を待つので 86 秒で戻る） |
| 2 | `--mode text` で入力 → 応答 → VOICEVOX 再生 | **応答までは OK、再生は NG**（PortAudio が無い）。VOICEVOX の合成は OK。合成した WAV を `pw-play` に渡すとエラーなく終了した（PipeWire の出力経路は開ける。人が聞いて確かめてはいない） |
| 3 | `--mode voice` で内蔵マイクと 3 往復以上、回り込みなし | **未実施**（PortAudio が無い＋人間の確認項目）。代わりに疑似音声で 3 往復を確認（下記） |
| 4 | `pytest` / `mypy --strict src` / `ruff check` | **OK**：pytest 112 passed（＋ integration 1 passed）、mypy「no issues found in 14 source files」、ruff「All checks passed!」 |
| 5 | `nvidia-smi` 実使用量、体感レイテンシ（ストップウォッチ 3 回） | VRAM は記録済み（下記）。**ストップウォッチ計測は人間が戻ってから**。代わりにプログラム内での計測値を記録した |
| 6 | push、コミットハッシュ報告 | 本ファイルを含むコミットを push 済み（ハッシュは最後のチャット報告に記載） |

### 疑似音声モードの結果（完了条件 3 の代替）

```
uv run python -m src.main --mode voice \
  --input-wav tests/fixtures/test_speech_ja.wav tests/fixtures/test_context_1.wav tests/fixtures/test_context_2.wav \
  --output-wav-dir <出力先>
```

| 往復 | 聞き取り（ASR、検出言語 ja） | ASR 完了 | LLM 完了 | 音声準備完了 |
|---|---|---|---|---|
| 1 | これは H S R のレイテンシ測定用のテスト音声です。マーケティング部の田中です。 | 0.42s | 1.28s | 3.25s |
| 2 | 売上は前年比百二十パーセントを達成しました。 | 0.32s | 1.07s | 3.52s |
| 3 | 展開率について報告します。現在、全社の八十五パーセントまで導入が完了。 | 0.38s | 1.01s | 3.01s |

- 時間は **VAD が発話終了と判定した時点**から測った（実際の発話終了から `min_silence_ms`＝0.4 秒遅れる）。体感値は、ここにだいたい 0.4 秒を足したものになる。
- 内訳: ASR 0.3〜0.4 秒、LLM 0.6〜0.9 秒（応答は 40〜65 トークン）、**VOICEVOX（CPU）の全文合成に 1.5〜2.5 秒**。いちばん大きいのは TTS（「4. 次にやるとよいこと」参照）。
- 往復ごとに応答 WAV（reply_01〜03.wav）ができている。自己発話ゲートの mute → 0.7 秒後の unmute もログで確認した。
- テスト音声は townhall の会議用なので、内容は井戸端の想定とは合っていない。ペルソナどおり 2〜3 文で、思考タグ（`<think>`）は出ていない。

text モードの応答例（LLM 0.5〜1.1 秒）:

> 入力: 最近、畑のナスがよく採れるんですよ。
> 応答: そうなんですね。ナスがよく採れるというのは、きっととても嬉しいことですね。お手入れがうまくいっているんでしょうね。

> 入力: 住民票を取るには何が必要ですか？
> 応答: 住民票を取るには、まず役所の窓口で確認するのがおすすめです。住所や氏名などの情報が必要になるので、お手持ちの証明書や書類をご用意ください。（以下略）

### VRAM 実使用量（nvidia-smi、ASR・LLM の両方を起動し往復を流した後）

| プロセス | 割当（utilization） | 実使用 |
|---|---|---|
| ASR（Qwen3-ASR-1.7B、重み 3.9GiB、KV 0.84GiB ≒ 7,840 トークン） | 0.35（5.6GB） | **5,278 MiB** |
| LLM（Qwen3-8B-AWQ、重み 5.71GiB、KV 1.15GiB ≒ 8,336 トークン） | 0.45（7.2GB） | **7,354 MiB** |
| 合計（GPU 全体） | 0.80 | **12,658 / 16,384 MiB**（空き約 3.6GB） |

## 2. 決めたこと（自分で判断したもの）

### サービング

- **ASR の配分を 0.35 に上げた（1-9 該当）。**0.30（4.8GB）では、重み 3.9GiB と CUDA 文脈・プロファイリングで KV キャッシュの枠が残らない。並列数とエンコーダ予約を絞った後でも、4096 トークンに要る 0.44GiB に対して 0.06GiB しか残らなかった。0.35 で KV 0.84GiB（同時 1.9 本ぶん）になる。
- **vLLM の並列設定を 16GB 機向けに絞った**（配分は変えずに収めるため）:
  - ASR: `--max-num-seqs 4 --max-num-batched-tokens 1024 --limit-mm-per-prompt '{"audio": 1}'`
  - LLM: `--max-num-seqs 8 --max-num-batched-tokens 1024 --enforce-eager`。既定設定（256 並列＋torch.compile／CUDA graph）だと 0.45 で KV 枠が −0.09GiB になる。`--enforce-eager` でようやく 8192 トークン 1 本ぶん（1.15GiB）が入った。eager のため decode はやや遅くなるはずだが、実測で LLM 応答 0.5〜1.1 秒なので問題にならない。
  - 窓口 1 か所・同時 1 会話という前提での設定。複数のエッジを 1 台で受けるときは見直しが要る。
- **llm は asr が healthy になってから起動する**（`depends_on: service_healthy`）。2 つの vLLM が同時にプロファイリングすると空き VRAM の見積もりが競合するため。
- ポートは `127.0.0.1` だけで待ち受ける（LAN へ出さない）。site プロファイルでエッジから受けるときに WP01 で見直す。
- vLLM イメージは townhall の Dockerfile をそのまま使い、`idobata-vllm:v0.27.1` として 1 回だけビルドして asr・llm で共有する。
- VOICEVOX は `cpu-latest`（0.25.2）。

### コード

- `edge_bridge.py` は `unity_bridge.py` をファイル名だけ変えたもの（中身・クラス名 `UnityBridge` もそのまま。改名は WP01 で拡張するときに行う）。
- `caption_feed.py` は townhall の `session_manager`（持ち込まない対象）を import していて、そのままでは読み込めない。そこで依存を `CaptionSession` プロトコル（`chat_active` / `suppress_captions` / `is_recording` / `handle_chat_utterance` / `accumulate_transcript`）に置き換えた。「参照名だけ直す」よりは少し踏み込んだ変更。テスト（`test_caption_feed_unit.py`）も合わせて直した。
- `schemas.py` は「そのまま」の指示だが、`TranscriptChunk` に `language: str = "ja"` を 1 フィールド足した（1-5 のため）。議事録用のスキーマは残してある（WP03 で紹介状スキーマに置き換える想定）。
- `tts_client.py` には `play(wav_bytes)` を足しただけ（合成と再生を分け、「音声開始」を正しく測るため）。
- `config.py` は `deploy/standalone/.env` を自前で読む（python-dotenv は足していない）。既にある環境変数のほうが優先。`IDOBATA_ENV_FILE` で別の .env を指定できる。
- LLM の `max_tokens` は既定 512、dialogue は 256、`temperature` 0.7。思考モードは `extra_body.chat_template_kwargs.enable_thinking=false` で切っている（サーバ側に `--reasoning-parser` は付けていない）。
- 日本語以外の発話は、ユーザー発話の頭に `[language: en]` を付けて LLM に渡す。システムプロンプトには「日本語以外ならその言語で短く答える」を入れてある。ただし TTS は VOICEVOX しかないので、英語などの応答は読み上げが不自然になる（多言語 TTS は P2）。
- ログ: 発話・応答の本文は `--debug` のときだけログに出す。CLI 画面への表示（`[聞き取り]` `[井戸端AI]`）は print（ユーザー向け出力）で、ディスクには何も書かない。
- 疑似音声モード: `--input-wav` は複数ファイルを受け付け、1 ファイル＝1 往復として順に流す。各ファイルの後に 1 秒の無音を足して VAD に終わりを判定させる。`--output-wav-dir` を付けると、再生する代わりに応答音声を WAV で書き出す（**AI 側の合成音声だけ**で、来訪者の音声は保存しない）。
- 拠点の語彙（`server/context/etajima.yaml` / `akiota.yaml`）は**記憶にもとづく仮置き**。役所・支所名、港・観光地・施設名、課名を各 10 語。課名は特に推測が多い。
- Python は `server/.python-version` で 3.12 に固定した（uv が 3.12.15 を使う）。依存は townhall と同じ（fastapi / uvicorn も残した）。

### テストデータ

- townhall のテスト用 WAV は Git LFS 管理で、`GIT_LFS_SKIP_SMUDGE=1` で clone したためポインタしか無かった。git-lfs も入っていない。そこで GitHub の LFS batch API から 4 本だけ取得し（SHA-256 一致を確認）、**idobata-ai には LFS を使わず普通のファイルとして**入れた（計 約 900KB）。

## 3. 人間に確認してほしいこと

1. **sudo が要る作業（1 行）**: Python から音声デバイスを開くために PortAudio を入れる。
   ```bash
   sudo apt install libportaudio2
   ```
   入れたら `cd ~/idobata-ai/server && uv run python -m src.main --list-devices` で「内部オーディオ」の入出力が出ることを確認する。PipeWire 側（`wpctl status`）では Sink 60 / Source 61「内部オーディオ アナログステレオ」が既定として見えている。
   - 任意: townhall の LFS ファイルを普通に扱いたい場合は `sudo apt install git-lfs`（今回は不要だった）。
2. **VRAM 配分の変更を承認するか**: ASR 0.35 ＋ LLM 0.45 ＝ 0.80。予備は **0.20（約 3.2GB、実際の空きは約 3.6GB）** になった。YOLO11-pose（n/s なら 1GB 未満）は入る見込み。多言語 TTS（Qwen3-TTS 0.6B）まで同居させると厳しいかもしれない。代わりの案:
   - (a) ASR を Qwen3-ASR-**0.6B** に落とす（0.30 以下に収まる見込み。日本語の精度は要確認）
   - (b) LLM の `--max-model-len` を 4096 にして 0.40 に下げる（会話 6 ターン＋システムプロンプトなら 1,000 トークン未満なので足りる）
   - (c) このまま（0.80）
3. **拠点の語彙 YAML の中身**（`server/context/*.yaml`）は仮。市・町の正式な課名と主要施設名に差し替えてほしい。
4. **`server/tests/fixtures/*.wav` を普通のファイルとしてコミットしたこと**（townhall の会議用音声。社名・人名など社外秘に当たる内容が無いかを確認してほしい。リポジトリの公開・非公開に注意）。
5. ペルソナの話しぶり: 相手に質問を返しすぎる傾向がある（「詳しく教えていただけますか？」が多い）。プロンプトの調整は WP03 の対話FSMと合わせてやるのがよさそう。

## 4. 次に人間がやる手順（音声モードの試し方）

```bash
# 0) 一度だけ: PortAudio
sudo apt install libportaudio2

# 1) サービング（起動済みなら不要。自動再起動 restart: unless-stopped）
cd ~/idobata-ai
docker compose -f deploy/standalone/docker-compose.yaml --env-file deploy/standalone/.env up -d
bash scripts/healthcheck.sh            # 3/3 OK を確認（起動直後は 2〜3 分かかる）

# 2) デバイス確認
cd server
uv run python -m src.main --list-devices

# 3) text モード（完了条件 2：再生まで）
uv run python -m src.main --mode text
#   > こんにちは      → 応答が表示され、スピーカーから VOICEVOX（四国めたん）の声が出る
#   /reset で履歴消去、/quit で終了

# 4) voice モード（完了条件 3：3 往復以上、回り込みなし）
uv run python -m src.main --mode voice
#   既定デバイスで駄目なら: --audio-device <入力ID> --output-device <出力ID>
#   1 往復ごとに「[計測] 発話終了→ ASR x.xxs / LLM x.xxs / 音声開始 x.xxs」が出る
#   内蔵マイク＋内蔵スピーカーは回り込みやすい。応答の終わりから 0.7 秒は聞かない（SpeechGate）

# 5) 完了条件 5：ストップウォッチで「話し終わり → 声が出始める」を 3 回測り、本ファイルに追記する

# 6) 無人の確認をやり直すとき
uv run pytest                          # unit（112 件）
uv run pytest -m integration           # 実サービス往復（サービング起動中のみ）
uv run python -m src.main --mode voice --input-wav tests/fixtures/test_speech_ja.wav --output-wav-dir /tmp/idobata-out
```

止めるとき: `docker compose -f deploy/standalone/docker-compose.yaml down`（GPU を約 12.6GB 使っているため、他の GPU 作業をする前に止める）。

## 5. 次にやるとよいこと（WP01 以降へのメモ、未着手）

- **TTS を文ごとに分けて合成・再生する**: 今は応答全文を合成してから鳴らしているため、発話終了から声が出るまでの 3 秒前後のうち 1.5〜2.5 秒が VOICEVOX（CPU）の合成。最初の 1 文だけ先に合成すれば 1.5 秒前後まで縮む見込み。LLM のストリーミングと組み合わせればさらに縮む。
- VOICEVOX の GPU 版は VRAM の予備を食うので、まず文分割を試す。
- vLLM の `--enforce-eager` を外したいときは、torch.compile のキャッシュ（`/root/.cache/vllm`）もボリュームに逃がす（再起動のたびに約 27 秒コンパイルが走るのを防ぐ）。
- ASR は「H S R」（正しくは ASR）のような略語の取り違えがある。townhall と同じく、固有名詞は LLM 側で補正する方針でよいと考える。
