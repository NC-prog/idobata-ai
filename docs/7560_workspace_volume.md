# 7560 — /workspace ボリュームの切り出し手順（保留中）

2026-10-19 時点のディスク構成:

```
nvme1n1 (WD_BLACK SN7100 1TB)  Ubuntu
├─p1 487M  vfat  /boot/efi
├─p2 186G  ext4  /        （空き 153G）
├─p3 714G  ext4  /home    （空き 566G）
└─p4  30G  swap
nvme0n1 (Dell PC SN810 2TB)    Windows
├─p1 100M  vfat  EFI system partition
├─p2  16M        Microsoft reserved
├─p3 1906G ntfs  Windows (C:)
└─p4   2G  ntfs  Recovery
```

当面は `/home` の空きで運用し、`HF_CACHE_DIR=$HOME/.cache/huggingface`、データは `$HOME/idobata-data`。
モデルやデータが増えて足りなくなったら以下を実施する。

## 1. Windows 側で C: を縮小

1. 設定 → システム → 電源 → 高速スタートアップをオフ。デバイスの暗号化が有効なら一時停止。
2. `diskmgmt.msc` → C: → ボリュームの縮小。Windows に 200GB 程度残す（最大値が 1,700,000MB 前後なら、そのまま）。
3. 未割り当てはフォーマットせず、完全シャットダウン。

縮小可能な最大値が小さい場合は、末尾にページファイル・休止ファイル・システムの復元があるのが原因。

## 2. Ubuntu 側でパーティション作成（新しい番号は p5）

```bash
sudo parted /dev/nvme0n1 unit GiB print free      # p3 の終了 = 新しい空きの開始を確認
sudo parted /dev/nvme0n1 --align optimal mkpart workspace ext4 <開始>GiB 1906GiB
sudo partprobe /dev/nvme0n1
sudo mkfs.ext4 -L workspace /dev/nvme0n1p5
sudo mkdir -p /workspace
UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p5)
echo "UUID=$UUID  /workspace  ext4  defaults,noatime,nofail  0 2" | sudo tee -a /etc/fstab
sudo systemctl daemon-reload
sudo mount -a
sudo chown fuji:fuji /workspace
mkdir -p /workspace/hf-cache /workspace/idobata-data
```

`nofail` を付けるので、このパーティションが見えないブートでも Ubuntu は起動する。
切り出し後は `.env` の `HF_CACHE_DIR=/workspace/hf-cache` に変え、既存キャッシュを `mv` で移す。
