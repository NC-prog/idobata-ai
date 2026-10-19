# テスト音声（fixtures）

来訪者役の日本語発話を VOICEVOX で合成したもの。実在の人の録音は含まない。
再生成: `uv run --project server python scripts/gen_test_audio.py`（VOICEVOX 起動中に）

| ファイル | 内容 | 話者 |
|---|---|---|
| `visitor_greeting_ja.wav` | 挨拶「こんにちは。ちょっと寄ってみました。今日は少し時間があるんです。」 | VOICEVOX:春日部つむぎ |
| `visitor_procedure_ja.wav` | 手続きの質問「住民票の写しを取りたいんですけど、どこの窓口に行けばいいですか。」 | VOICEVOX:玄野武宏 |
| `visitor_smalltalk_ja.wav` | 雑談「最近、急に寒くなりましたね。朝晩は特に冷えますよ。」 | VOICEVOX:青山龍星 |

形式: 24kHz / mono / PCM16、前 0.5 秒・後 0.8 秒の無音つき。
井戸端AI 側の声（四国めたん）と区別するため、来訪者役は別の話者にしている。
