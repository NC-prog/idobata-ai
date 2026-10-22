# 別機エッジ（Windows ミニ PC）の設置設計と立ち上げ手順（WP02 D）

窓口のロビーに置くのは **エッジ PC**（画面・カメラ・マイク・スピーカー）だけ。会話の処理（VAD・ASR・LLM・VOICEVOX・姿勢推定・話者の区別）は
**フロント AI 機**（7560、のちに T2）で行う。エッジ PC は Chrome でフロントのページを開くだけで、**何も保存しない**（音声・映像・会話）。

## 1. 構成

```
 [ロビー]                                              [事務室・ラック]
 縦置きモニタ ── HDMI ──┐
 USB カメラ（画角78〜90°）─ USB ─┤ エッジ PC                        フロント AI 機
 AEC 付き USB スピーカーフォン ─ USB ─┤ Windows 11 / 内蔵 GPU ── LAN ── スイッチングハブ ── LAN ── 7560（GPU）
                                  └ Chrome（キオスク）                                   │
                                                                                         └ （ほかの機器はつながない）
```

| 機器 | 要件 | 理由 |
|---|---|---|
| エッジ PC | Windows 11、**内蔵 GPU で WebGL がハードウェア描画できること**（Intel UHD 以上）、メモリ 8GB、有線 LAN | VRM アバターの描画。ソフトウェア描画だとメインスレッドが埋まり、カメラの送出が 0.7fps まで落ちた（WP02 A の測定）。7560 の Intel UHD（TGL GT1）では描画 61fps・送出 5.0fps |
| USB カメラ | 画角 78〜90°、720p、UVC（ドライバ不要） | 輪の判定（距離＝矩形の高さの割合）は画角に依存する。替えたら `party.min_box_ratio` を測り直す（6 章） |
| スピーカーフォン | **AEC（エコーキャンセル）付き**、UVC/UAC（ドライバ不要） | アバターの声をマイクが拾わない。フロントの自己発話ゲートと二重 |
| モニタ | 縦置き（9:16）、来訪者の目の高さ | |
| ハブ | ギガビットのスイッチングハブ。**エッジ PC とフロントだけ**をつなぐ（庁内 LAN・インターネットには出さない） | 「会話・音声・映像は建物の外に出さない」。帯域は上り約 0.1Mbps（音声 32kB/s＋JPEG 約 10kB×5fps）、下りは応答の WAV |

### IP の例（固定）

| 機器 | IP | 備考 |
|---|---|---|
| フロント AI 機 | `192.168.50.10/24` | ゲートウェイなし（直結の島） |
| エッジ PC | `192.168.50.21/24` | ゲートウェイなし |

（例の値。庁内の割り当てに合わせて決める。フロントが庁内 LAN にも足を持つときは、エッジ側の NIC だけこの島にする）

## 2. つながり方

1. エッジ PC の Chrome が `http://192.168.50.10:8080/edge/` を開く（ページの配信はフロント。エッジ PC には何も入れない）
2. ページがフロントの WebSocket `ws://192.168.50.10:8765` につなぐ（`?ws=` でポートを変えられる）。1 本の接続で、
   上り＝マイクの PCM（種別 0x01）とカメラの JPEG（種別 0x02、320×180・5fps）、`hello` など、下り＝応答の文と WAV・口形・字幕・輪の状態
3. フロントは `IDOBATA_EDGE_ALLOW` の IP からだけ `/edge/` と WebSocket を受け付ける（それ以外は 403 と close 4003）。
   職員パネル `/staff/`・管理者ページ `/admin/` はこれとは別（LAN 内＋パスワード）

## 3. ブラウザのカメラ・マイク（secure context）の方式

Chrome は `http://localhost` 以外の `http://` のページでは `getUserMedia`（カメラ・マイク）を使わせない（secure context が要る）。

| 方式 | 内容 | 長所 | 短所 |
|---|---|---|---|
| ① エッジ PC でローカル配信 | エッジ PC に小さな静的配信（または reverse proxy）を入れ、`http://localhost:8080/edge/` を開く。WebSocket だけフロントへ | 証明書もフラグも要らない | エッジ PC にサーバと、ページの更新の仕組み（フロントから配る）が要る。エッジ PC が「何も入れない端末」でなくなる |
| **② 安全でないオリジンを secure とみなす**（推奨） | Chrome の起動に `--unsafely-treat-insecure-origin-as-secure=http://192.168.50.10:8080` を付ける（同じことを Chrome のポリシー `OverrideSecurityRestrictionsOnInsecureOrigin` でもできる） | **設定 1 行、証明書の管理なし**。ページの更新はフロントだけ | 指定したオリジンだけ secure 扱い（閉じた島の中なので問題は小さい）。起動フラグ版は「サポートされていないフラグ」の警告バーが出ることがある（ポリシー版なら出ない） |
| ③ 自己署名証明書 | フロントを `https://` にし、エッジ PC に証明書（CA）を入れる | 正攻法 | 証明書の作成・配布・期限切れの管理が要る。期限切れでロビーの画面が止まる |

**② を推奨**。閉じたネットワーク（エッジ PC とフロントだけのハブ）の中で、指定したオリジン 1 つだけを secure とみなすので、
緩める範囲が最小で、運用で手を入れるものが無い。**起動フラグより、Chrome のポリシー（レジストリ）で入れる**ほうがよい
（警告バーが出ない・ショートカットを誰かが直しても効く）。

確認（7560 の上で、LAN の IP でページを開いて代用）: Chromium にフラグ無し → `isSecureContext: false`・`navigator.mediaDevices` 無し。
フラグあり → `isSecureContext: true`・`mediaDevices` あり、1 往復（ASR 0.43 秒・音声開始 0.43 秒）。
※ Playwright の「headless shell」はこのフラグを無視する（通常の Chromium・Chrome では効く）。

### Chrome のポリシー（エッジ PC、管理者の PowerShell）

```powershell
$p = "HKLM:\SOFTWARE\Policies\Google\Chrome"
New-Item -Path $p -Force | Out-Null
# ② フロントのページを secure とみなす
New-Item -Path "$p\OverrideSecurityRestrictionsOnInsecureOrigin" -Force | Out-Null
Set-ItemProperty -Path "$p\OverrideSecurityRestrictionsOnInsecureOrigin" -Name "1" -Value "http://192.168.50.10:8080"
# カメラ・マイクの許可ダイアログを出さない（このオリジンだけ）
New-Item -Path "$p\VideoCaptureAllowedUrls" -Force | Out-Null
Set-ItemProperty -Path "$p\VideoCaptureAllowedUrls" -Name "1" -Value "http://192.168.50.10:8080"
New-Item -Path "$p\AudioCaptureAllowedUrls" -Force | Out-Null
Set-ItemProperty -Path "$p\AudioCaptureAllowedUrls" -Name "1" -Value "http://192.168.50.10:8080"
# 音声の自動再生を許す（このオリジンだけ）
New-Item -Path "$p\AutoplayAllowlist" -Force | Out-Null
Set-ItemProperty -Path "$p\AutoplayAllowlist" -Name "1" -Value "http://192.168.50.10:8080"
# 翻訳・パスワード保存・ログインの案内を出さない
Set-ItemProperty -Path $p -Name "TranslateEnabled" -Value 0 -Type DWord
Set-ItemProperty -Path $p -Name "PasswordManagerEnabled" -Value 0 -Type DWord
Set-ItemProperty -Path $p -Name "BrowserSignin" -Value 0 -Type DWord
# ハードウェアアクセラレーション（WebGL を GPU で）
Set-ItemProperty -Path $p -Name "HardwareAccelerationModeEnabled" -Value 1 -Type DWord
```

`chrome://policy` で読み込まれたことを確かめる。

## 4. キオスク起動のショートカット

`C:\idobata\kiosk.cmd`:

```bat
@echo off
rem 井戸端AI エッジ（キオスク）。フロントの IP とポートは設置先に合わせる
set FRONT=http://192.168.50.10:8080
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --kiosk "%FRONT%/edge/" ^
  --user-data-dir=C:\idobata\chrome-profile ^
  --autoplay-policy=no-user-gesture-required ^
  --use-fake-ui-for-media-stream ^
  --unsafely-treat-insecure-origin-as-secure=%FRONT% ^
  --no-first-run --disable-session-crashed-bubble --disable-features=Translate ^
  --overscroll-history-navigation=0 --disable-pinch
```

- `--use-fake-ui-for-media-stream` はカメラ・マイクの許可ダイアログを省く（3 章のポリシーを入れたなら無くてもよい）
- `--unsafely-treat-insecure-origin-as-secure` は 3 章のポリシーと同じ。ポリシーを入れたら外してよい（警告バーが出なくなる）
- 専用のプロファイル（`--user-data-dir`）にする（ふだんの Chrome と混ざらない・キオスクの設定が固定される）
- 複数のカメラがあるときは、フロントの `IDOBATA_CAMERA_LABEL` に USB カメラの名前の一部を入れる（ページがその名前のカメラを選ぶ）
- キオスクを抜けるのは `Alt+F4`（保守のとき）

## 5. Windows の設定

| 項目 | 設定 | 場所 |
|---|---|---|
| スリープ・画面オフ | 「なし」（電源接続時） | 設定 → システム → 電源 |
| 自動ログオン | 専用のローカルアカウント（管理者でない）で自動ログオン | Sysinternals の Autologon（パスワードは LSA に暗号化して保存）。`netplwiz` でも可 |
| 起動時にキオスク | `kiosk.cmd` のショートカットを `shell:startup` に置く | |
| Windows Update | アクティブ時間を 7:00〜20:00、再起動は夜間（例 3:00）。ロビーの開庁時間に再起動しない | 設定 → Windows Update → 詳細オプション |
| 停電・電源断からの復帰 | BIOS の「AC 電源復帰時: 電源オン（Power On / Last State）」 | BIOS（機種ごと） |
| 通知 | 応答不可（集中モード）、通知をすべてオフ | 設定 → システム → 通知 |
| 音声 | 既定の入出力をスピーカーフォンに。マイクの「拡張」はオフ（Chrome の AEC と二重にしない） | 設定 → システム → サウンド |
| カメラ・マイクのプライバシー | 「デスクトップ アプリにカメラ／マイクへのアクセスを許可する」をオン | 設定 → プライバシーとセキュリティ |
| ネットワーク | エッジ側 NIC を固定 IP（1 章）、ネットワークの種類は「プライベート」 | |
| 画面の向き | 縦（モニタに合わせて） | 設定 → システム → ディスプレイ |

## 6. フロント AI 機の設定（`deploy/standalone/.env`）

```
IDOBATA_EDGE_HTTP_HOST=0.0.0.0        # ページを LAN に出す
IDOBATA_EDGE_WS_HOST=0.0.0.0          # WebSocket を LAN に出す
IDOBATA_EDGE_ALLOW=192.168.50.21      # エッジ PC の IP だけ（カンマ区切りで複数、CIDR も可）。この機体自身はいつでも可
IDOBATA_CAMERA=1
IDOBATA_CAMERA_LABEL=<USB カメラの名前の一部>
```

- ファイアウォールでも 8080・8765 をエッジ PC の IP だけに開ける（二重の網）。Ubuntu の ufw なら
  `sudo ufw allow from 192.168.50.21 to any port 8080,8765 proto tcp`（sudo が要るので、設置作業の担当が行う）
- カメラを外付け（画角 78〜90°）に替えたら、**`party.min_box_ratio` を測り直す**: `--camera-debug` で起動し、来訪者役が
  カメラから 2〜2.5m に立って `http://<フロント>:8080/edge/debug/party` の `people` の 1 つ目の値（矩形の高さの割合）を読む。
  その値を `server/src/config.py` の `PartyConfig.min_box_ratio` に入れる（7560 内蔵の既定は 0.35）

## 7. 箱から出して 30 分で立ち上げる手順

前提: フロント AI 機は動いている（`bash scripts/healthcheck.sh` が 3/3 OK）。エッジ PC には Windows 11 が入っている。

| 時間 | 作業 |
|---|---|
| 0〜5 分 | 配線: モニタ（HDMI）・USB カメラ・スピーカーフォン・LAN（ハブ）・電源。カメラはモニタの上の中央、来訪者の顔の高さに向ける |
| 5〜10 分 | Windows の初期設定（ローカルアカウント `idobata-kiosk`）。固定 IP（1 章）。`ping 192.168.50.10` が通ること |
| 10〜15 分 | Chrome を入れる（オフラインインストーラを USB メモリで）。3 章のポリシーを PowerShell で入れ、`chrome://policy` で確認 |
| 15〜18 分 | `C:\idobata\kiosk.cmd`（4 章）を置き、`shell:startup` にショートカット |
| 18 分 | **GPU で描画しているかを確かめる**: キオスクではない普通の Chrome（同じプロファイル `--user-data-dir=C:\idobata\chrome-profile`）で `chrome://gpu` を開き、「Graphics Feature Status」の **WebGL** と **WebGL2** が「**Hardware accelerated**」であること。「Software only」「Unavailable」なら、GPU のドライバを入れ直す（ソフトウェア描画ではカメラの送出が 1fps 未満に落ちる。WP02 A の測定） |
| 18〜23 分 | 5 章の Windows の設定（電源・自動ログオン・Update の時間・通知・音声の既定・カメラとマイクの許可・画面の向き） |
| 23〜25 分 | フロントの `.env` に 6 章の設定を入れて再起動（`IDOBATA_EDGE_ALLOW` にエッジ PC の IP） |
| 25〜30 分 | `kiosk.cmd` を実行 → アバターが出る → フロントのログに `エッジ hello` と `エッジのカメラ: ok=True label=<USB カメラ>`。カメラの前に立って 2 秒で `輪: PRESENT → PARTY` と切り出しの声、話しかけて応答、離れて別れの一言。最後に再起動して、自動ログオン → キオスク → つながるまでを確かめる |

### うまくいかないとき

| 症状 | 見るところ |
|---|---|
| 「マイクを開けません」「カメラを開けません」 | `chrome://policy` に 3 章のポリシーがあるか。無ければ起動フラグの `--unsafely-treat-insecure-origin-as-secure` の URL がページの URL と完全に同じか（ポート込み） |
| つながらない（未接続のまま） | フロントの `IDOBATA_EDGE_ALLOW`・`IDOBATA_EDGE_WS_HOST=0.0.0.0`、ファイアウォール。フロントのログに `許可していない相手からのエッジ接続を閉じました` が出ていれば ALLOW の IP 違い |
| アバターがカクつく・輪の反応が遅い | `chrome://gpu` で WebGL が「Hardware accelerated」か。ソフトウェア描画ならドライバを入れ直す（カメラの送出が 1fps 未満に落ちる） |
| 違うカメラが使われる | フロントの `IDOBATA_CAMERA_LABEL`。ログの `エッジのカメラ: … label=` の名前を見て、その一部を入れる |
| アバターの声をマイクが拾う | スピーカーフォンの AEC、Windows のマイクの「拡張」をオフ、音量を下げる |

## 8. 実測（別機）

手元に Windows 機が無かったので、**別機での実測はまだ**。7560 の上で、LAN の IP でページを開いて別機の経路（0.0.0.0 での待ち受け・
許可した IP だけ・方式②・1 往復）を確かめた（`docs/WP02_result.md` の D）。
