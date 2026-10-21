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
| `visitor_trouble_ja.wav` | 課の決まらない相談「ちょっと困ったことがあって。」（WP04 未特定） | VOICEVOX:青山龍星 |

形式: 24kHz / mono / PCM16、前 0.5 秒・後 0.8 秒の無音つき。
井戸端AI 側の声（四国めたん）と区別するため、来訪者役は別の話者にしている。
