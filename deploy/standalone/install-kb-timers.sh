#!/usr/bin/env bash
# 知識パックの自動更新を systemd の user タイマーで有効にする:
#   毎朝 6:00 更新の速い区画（--daily）/ 毎週日曜 3:00 全体の作り直し / 毎時 05 分 天気
#   パックの更新の後に、手続きナレッジの出典の再確認（scripts/check_procedures.py、WP09）
# sudo は使わない。使い方: bash deploy/standalone/install-kb-timers.sh [拠点]（省略時は .env の IDOBATA_SITE）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$HERE/.env"
val() { [ -f "$ENV_FILE" ] && grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d "\"'" || true; }
SITE="${1:-$(val IDOBATA_SITE)}"; SITE="${SITE:-etajima}"
DATA="$(eval echo "$(val IDOBATA_DATA_DIR)")"; DATA="${DATA:-$HOME/idobata-data}"
UV="$(command -v uv)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR" "$DATA/kb"
# 以前の版（毎朝の全体の作り直し）のタイマーが残っていれば外す
if systemctl --user is-enabled "kb-nightly@$SITE.timer" >/dev/null 2>&1; then
  systemctl --user disable --now "kb-nightly@$SITE.timer" || true
fi
rm -f "$UNIT_DIR/kb-nightly@.service" "$UNIT_DIR/kb-nightly@.timer"
for f in kb-daily@.service kb-daily@.timer kb-full@.service kb-full@.timer kb-weather@.service kb-weather@.timer; do
  sed -e "s|__UV__|$UV|g" -e "s|__REPO__|$REPO|g" -e "s|__DATA__|$DATA|g" "$HERE/$f" > "$UNIT_DIR/$f"
done
systemctl --user daemon-reload
systemctl --user enable --now "kb-daily@$SITE.timer" "kb-full@$SITE.timer" "kb-weather@$SITE.timer"
systemctl --user list-timers "kb-*@$SITE.timer" --no-pager

# ログアウト中・再起動後（ログイン前）も動かすには linger が要る
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  if loginctl enable-linger "$USER" 2>/dev/null; then
    echo "linger を有効にしました（ログアウト中もタイマーが動きます）"
  else
    echo "【人間の操作が要ります】linger を有効にできませんでした。ログアウト中や再起動直後（ログイン前）も更新するには:"
    echo "  sudo loginctl enable-linger $USER"
  fi
fi
