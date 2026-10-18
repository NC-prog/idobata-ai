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

富士製作所合同会社 / 広島オープンアクセラレータ 2026（The Meet 2026）
