# テスト音声（fixtures）

来訪者役の日本語発話を VOICEVOX で合成したもの。実在の人の録音は含まない。
再生成: `uv run --project server python scripts/make_fixtures.py`（VOICEVOX 起動中に）

| ファイル | 内容 | 話者 |
|---|---|---|
| `visitor_greeting_ja.wav` | 挨拶「こんにちは。ちょっと寄ってみました。今日は少し時間があるんです。」 | VOICEVOX:春日部つむぎ |
| `visitor_procedure_ja.wav` | 手続きの質問「住民票の写しを取りたいんですけど、どこの窓口に行けばいいですか。」 | VOICEVOX:玄野武宏 |
| `visitor_smalltalk_ja.wav` | 雑談「最近、急に寒くなりましたね。朝晩は特に冷えますよ。」 | VOICEVOX:青山龍星 |
| `visitor_event_ja.wav` | 行事の質問「今週末は道の駅で何かやってますか。」（WP04 調べる） | VOICEVOX:冥鳴ひまり |
| `visitor_yes_ja.wav` | 承諾「うん。」 | VOICEVOX:冥鳴ひまり |
| `visitor_no_ja.wav` | 辞退「いらない。」 | VOICEVOX:冥鳴ひまり |
| `visitor_kokuho_ja.wav` | 手続きの相談「国保の切り替えで聞きたいことがあって。」（WP04 つなぐ） | VOICEVOX:玄野武宏 |
| `visitor_onegai_ja.wav` | 承諾「お願いします。」 | VOICEVOX:玄野武宏 |
| `visitor_intake_reason_ja.wav` / `visitor_intake_when_ja.wav` / `visitor_intake_self_ja.wav` | 聞き取りの答え「会社を退職したので。」「来月からです。」「本人です。」（WP04） | VOICEVOX:玄野武宏 |
| `visitor_name_ja.wav` / `visitor_hai_ja.wav` / `visitor_area_ja.wav` | 「山田です。」（架空の名前）「はい。」「能美町です。」 | VOICEVOX:玄野武宏 |
| `visitor_contact_ja.wav` / `visitor_phone_ja.wav` | 「電話でお願いします。」「090の1234の5678です。」（架空の番号。かなで読ませる） | VOICEVOX:玄野武宏 |
| `visitor_phone_part1_ja.wav` / `_part2_` / `_part3_` | 区切って言った番号の断片「ゼロキューゼロの。」「イチニーサンヨンの。」「ゴーロクナナハチ。」（WP05 実機の指摘。part3 は保存後に重なる断片の確認にも使う） | VOICEVOX:玄野武宏 |
| `visitor_phone_refuse_ja.wav` / `visitor_phone_notasked_ja.wav` | 「番号はちょっと言いたくないです。」「電話番号、聞いてないよね。」（WP07 連絡方法が電話で番号を言わない・保存の後に触れる） | VOICEVOX:玄野武宏 |
| `pair_honin_{1,2,3}_ja.wav` / `pair_tsuki_{1,2,3}_ja.wav` | 本人と付き添いの会話（WP02 B の話者の区別。交互に流す） | VOICEVOX:玄野武宏（本人）・冥鳴ひまり（付き添い） |
| `staff_bg_{1,2}_ja.wav` | 後ろの職員の声（輪が無いときと、会話の途中に流して捨てられることを見る） | VOICEVOX:青山龍星 |
| `visitor_trouble_ja.wav` | 課の決まらない相談「ちょっと困ったことがあって。」（WP04 未特定） | VOICEVOX:青山龍星 |

形式: 24kHz / mono / PCM16、前 0.5 秒・後 0.8 秒の無音つき。
井戸端AI 側の声（四国めたん）と区別するため、来訪者役は別の話者にしている。
