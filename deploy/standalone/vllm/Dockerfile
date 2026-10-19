# Qwen3-ASR-1.7B 用 vLLM コンテナ
# ベースイメージには音声処理ライブラリが含まれないため、vllm[audio] を追加インストールする
FROM vllm/vllm-openai:v0.27.1

RUN uv pip install --system "vllm[audio]==0.27.1"
