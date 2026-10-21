# 井戸端AI（仮）

自治体の庁舎に置く、ローカルAIの多言語・マルチモーダル総合窓口アバター。
映像・音声・会話は建物の外に出ない。「相談窓口」ではなく「井戸端」——用がなくても居てもらう。

- 構想メモ: [docs/構想メモ.md](docs/構想メモ.md)
- 母体: [NC-prog/townhall-assistant](https://github.com/NC-prog/townhall-assistant)（Prudential向けタウンホール司会アシスタント）

## 構成（3層）

| 層 | 機器 | 役割 |
|---|---|---|
| エッジ | 既存Windows機（GPU不要） | カメラ・マイク・スピーカー・表示。ブラウザで VRM アバターを描画 |
| フロントAI | 現地GPU機（12〜24GB） | 人物検知（YOLO11-pose）、VAD、Qwen3-ASR、TTS、セッション |
| バックエンドAI | RTX PRO 6000 | LLM（gpt-oss）、VLM、RAG、紹介状生成、職員パネル |

## 起動（WP01：7560 単体、ブラウザのアバター）

```bash
cp deploy/standalone/.env.example deploy/standalone/.env   # 初回のみ
docker compose -f deploy/standalone/docker-compose.yaml --env-file deploy/standalone/.env up -d --build
bash scripts/healthcheck.sh

(cd edge && npm ci && npm run build)        # エッジページのビルド（初回と edge/ を変えたとき）

cd server
uv sync
uv run python -m src.main --mode voice --edge   # → Chrome で http://localhost:8080/edge/ を開き「はじめる」
# ローカルのマイク・スピーカーで動かす場合（デバイスは番号より名前の一部で指定）
uv run python -m src.main --list-devices
uv run python -m src.main --mode voice --audio-device hw:0,0 --output-device alsa_output.pci-0000_00_1f.3
```

### マイクについて

- **会議用マイク（Avaya B109 などのスピーカーフォン）は部屋の会話まで拾う**。話しかけていない人の声にも応答してしまうので、
  WP02 の声かけ機能（人を検知してから聞く）が入るまでは、入力の音量を 0.5 程度に下げて使う（sudo 不要）:
  ```bash
  wpctl status                          # Sources の中の B109 の番号を確認（例: 78. Avaya B109 モノ）
  wpctl set-volume <B109のsource id> 0.5
  ```
  番号は挿し直すと変わる。戻すときは `1.0`。
- 内蔵マイクは ALSA 直（`hw:0,0`）を使う。PipeWire 経由の入力はノイズになる（[docs/WP00_result.md](docs/WP00_result.md) の 6-2）。
  2026-10-21 時点では hw:0,0 もノイズになっていた（[docs/WP01_result.md](docs/WP01_result.md) A-4）。
- 最後の発話から 90 秒（`IDOBATA_IDLE_TIMEOUT_SEC`）経つと会話の履歴を消して待機に戻る。次の発話は新しい会話として扱う。

詳しい手順と開発ルールは [CLAUDE.md](CLAUDE.md)。

富士製作所合同会社 / 広島オープンアクセラレータ 2026（The Meet 2026）
