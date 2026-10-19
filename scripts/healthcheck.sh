#!/usr/bin/env bash
# healthcheck.sh — 井戸端AI 前提サービスの一括ヘルスチェック
#
# 使い方:
#   bash scripts/healthcheck.sh
#
# IDOBATA_PROFILE（standalone / site）に応じて確認先を切り替える。
#   standalone: ASR(8005) / LLM(8001) / VOICEVOX(50021) すべて 127.0.0.1
#   site      : ASR / VOICEVOX は 127.0.0.1、LLM は IDOBATA_BACKEND_URL
# deploy/standalone/.env があれば読み込む（既にある環境変数が優先）。
#
# 終了コード: 0=全必須サービスOK, 1=いずれかNG

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${IDOBATA_ENV_FILE:-${SCRIPT_DIR}/../deploy/standalone/.env}"

if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        case "$key" in ''|\#*) continue ;; esac
        if [ -z "${!key:-}" ]; then
            export "$key=$(eval echo "$value")"
        fi
    done < "$ENV_FILE"
fi

PROFILE="${IDOBATA_PROFILE:-standalone}"
HOST="127.0.0.1"
REQUIRED_OK=0
REQUIRED_TOTAL=0

check_vllm() {
    local name="$1"
    local base_url="$2"   # .../v1 まで
    local label="$3"

    REQUIRED_TOTAL=$((REQUIRED_TOTAL + 1))

    local resp
    resp=$(curl -s --connect-timeout 3 --max-time 5 "${base_url}/models" 2>/dev/null || echo "")

    if [ -n "$resp" ] && echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        local model
        model=$(echo "$resp" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('data', [])
print(models[0].get('id', '?') if models else '(empty)')
" 2>/dev/null || echo "?")
        if [ "$model" = "$label" ]; then
            printf "[OK] %-12s (%s) — %s\n" "$name" "$base_url" "$model"
            REQUIRED_OK=$((REQUIRED_OK + 1))
        else
            printf "[NG] %-12s (%s) — モデル名が %s（期待値 %s）\n" "$name" "$base_url" "$model" "$label"
        fi
    else
        printf "[NG] %-12s (%s) — 応答なし\n" "$name" "$base_url"
    fi
}

check_voicevox() {
    REQUIRED_TOTAL=$((REQUIRED_TOTAL + 1))

    local resp
    resp=$(curl -s --connect-timeout 3 --max-time 5 "http://${HOST}:50021/version" 2>/dev/null || echo "")

    if [ -n "$resp" ]; then
        printf "[OK] %-12s (%s) — version %s\n" "VOICEVOX" "http://${HOST}:50021" "$(echo "$resp" | tr -d '"')"
        REQUIRED_OK=$((REQUIRED_OK + 1))
    else
        printf "[NG] %-12s (%s) — 応答なし\n" "VOICEVOX" "http://${HOST}:50021"
    fi
}

echo "=== 井戸端AI ヘルスチェック（profile: ${PROFILE}） ==="

check_vllm "Qwen3-ASR" "http://${HOST}:8005/v1" "/models/asr"
case "$PROFILE" in
    standalone)
        check_vllm "LLM" "http://${HOST}:8001/v1" "/models/llm"
        ;;
    site)
        if [ -z "${IDOBATA_BACKEND_URL:-}" ]; then
            REQUIRED_TOTAL=$((REQUIRED_TOTAL + 1))
            echo "[NG] LLM          — IDOBATA_BACKEND_URL が未設定"
        else
            check_vllm "LLM" "${IDOBATA_BACKEND_URL%/}" "/models/llm"
        fi
        ;;
    *)
        echo "[NG] 不明なプロファイル: ${PROFILE}"
        exit 1
        ;;
esac
check_voicevox

# VRAM 表示（プロセス別）
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [ -n "$VRAM_INFO" ]; then
        printf "VRAM: %s / %s MiB\n" "$(echo "$VRAM_INFO" | awk -F', ' '{print $1}')" "$(echo "$VRAM_INFO" | awk -F', ' '{print $2}')"
    fi
fi

echo "=== 必須サービス: ${REQUIRED_OK}/${REQUIRED_TOTAL} OK ==="

if [ "$REQUIRED_OK" -lt "$REQUIRED_TOTAL" ]; then
    exit 1
fi
