"""井戸端AI のテスト音声（server/tests/fixtures/*.wav）を VOICEVOX で合成する。

来訪者役の発話（挨拶・手続きの質問・雑談、WP04 の行為の往復）を作る。会議音声など実在の録音は使わない。
応答側（井戸端AI）の声は四国めたん（speaker 2）なので、来訪者役は別の話者にしている。

使い方（VOICEVOX が 127.0.0.1:50021 で起動していること）:
    uv run --project server python scripts/make_fixtures.py
    uv run --project server python scripts/make_fixtures.py --out-dir /tmp/fixtures

出力は VOICEVOX 既定の 24kHz / mono / PCM16 WAV。前後に無音を付け、VAD が発話区間を切り出せるようにする。
音声の利用は VOICEVOX および各キャラクターの利用規約に従う（クレジットは fixtures/README.md）。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "server" / "tests" / "fixtures"
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"

TIMEOUT_SEC = 60.0
MAX_RETRIES = 3
RETRY_WAIT_SEC = 2.0
# VAD が発話の始まり・終わりを確定できるよう、前後に足す無音（秒）
PRE_SILENCE_SEC = 0.5
POST_SILENCE_SEC = 0.8


@dataclass(frozen=True)
class Utterance:
    """合成する 1 発話。"""

    filename: str
    speaker_id: int
    speaker_name: str
    text: str


UTTERANCES: tuple[Utterance, ...] = (
    Utterance(
        filename="visitor_greeting_ja.wav",
        speaker_id=8,
        speaker_name="春日部つむぎ",
        text="こんにちは。ちょっと寄ってみました。今日は少し時間があるんです。",
    ),
    Utterance(
        filename="visitor_procedure_ja.wav",
        speaker_id=11,
        speaker_name="玄野武宏",
        text="住民票の写しを取りたいんですけど、どこの窓口に行けばいいですか。",
    ),
    Utterance(
        filename="visitor_smalltalk_ja.wav",
        speaker_id=13,
        speaker_name="青山龍星",
        text="最近、急に寒くなりましたね。朝晩は特に冷えますよ。",
    ),
    # WP04: 同意付き行為（調べる・つなぐ）の往復。春日部つむぎの「うん」は ASR が聞き取れない
    # （「が」「がん」になる）ので、別の話者にしている
    Utterance(
        filename="visitor_event_ja.wav",
        speaker_id=14,
        speaker_name="冥鳴ひまり",
        text="今週末は道の駅で何かやってますか。",
    ),
    Utterance(filename="visitor_yes_ja.wav", speaker_id=14, speaker_name="冥鳴ひまり", text="うん。"),
    Utterance(filename="visitor_no_ja.wav", speaker_id=14, speaker_name="冥鳴ひまり", text="いらない。"),
    Utterance(
        filename="visitor_kokuho_ja.wav",
        speaker_id=11,
        speaker_name="玄野武宏",
        text="国保の切り替えで聞きたいことがあって。",
    ),
    Utterance(filename="visitor_onegai_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="お願いします。"),
    # WP04: 紹介状の聞き取り（同意の後）。名前・電話番号は架空
    Utterance(filename="visitor_intake_reason_ja.wav", speaker_id=11, speaker_name="玄野武宏",
              text="会社を退職したので。"),
    Utterance(filename="visitor_intake_when_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="来月からです。"),
    Utterance(filename="visitor_intake_self_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="本人です。"),
    Utterance(filename="visitor_name_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="山田です。"),
    Utterance(filename="visitor_hai_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="はい。"),
    Utterance(filename="visitor_area_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="能美町です。"),
    Utterance(filename="visitor_contact_ja.wav", speaker_id=11, speaker_name="玄野武宏",
              text="電話でお願いします。"),
    Utterance(filename="visitor_phone_ja.wav", speaker_id=11, speaker_name="玄野武宏",
              text="ゼロキューゼロの、イチニーサンヨンの、ゴーロクナナハチです。"),
    # 区切って言った番号を ASR が断片で返す場合（WP05 実機）。1 断片 1 ファイル。最後のものは保存後に重なる断片
    Utterance(filename="visitor_phone_part1_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="ゼロキューゼロの。"),
    Utterance(filename="visitor_phone_part2_ja.wav", speaker_id=11, speaker_name="玄野武宏",
              text="イチニーサンヨンの。"),
    Utterance(filename="visitor_phone_part3_ja.wav", speaker_id=11, speaker_name="玄野武宏",
              text="ゴーロクナナハチ。"),
    # 本人と付き添い（WP02 B の話者の区別。本人＝玄野武宏、付き添い＝冥鳴ひまり）と、後ろの職員の声（青山龍星）
    Utterance(filename="pair_honin_1_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="自分の介護のことで相談に来ました。"),
    Utterance(filename="pair_tsuki_1_ja.wav", speaker_id=14, speaker_name="冥鳴ひまり", text="私は娘で、付き添いで来ました。"),
    Utterance(filename="pair_honin_2_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="最近、足がずいぶん弱ってきましてね。"),
    Utterance(filename="pair_tsuki_2_ja.wav", speaker_id=14, speaker_name="冥鳴ひまり", text="家の中でも転びそうになるんです。"),
    Utterance(filename="pair_honin_3_ja.wav", speaker_id=11, speaker_name="玄野武宏", text="介護保険というのは、使えるんですかね。"),
    Utterance(filename="pair_tsuki_3_ja.wav", speaker_id=14, speaker_name="冥鳴ひまり", text="申請のしかたを知りたいんです。"),
    Utterance(filename="staff_bg_1_ja.wav", speaker_id=13, speaker_name="青山龍星", text="この書類、あとで課長に回しておいてください。"),
    Utterance(filename="staff_bg_2_ja.wav", speaker_id=13, speaker_name="青山龍星", text="コピー機の紙が切れてますよ、補充しておきますね。"),
    Utterance(
        filename="visitor_trouble_ja.wav",
        speaker_id=13,
        speaker_name="青山龍星",
        text="ちょっと困ったことがあって。",
    ),
)


def _post_with_retry(
    client: httpx.Client, path: str, params: dict[str, Any], body: Any = None
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.post(path, params=params, json=body)
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            print(f"  {path} 失敗（{attempt}/{MAX_RETRIES}）: {exc}", file=sys.stderr)
            time.sleep(RETRY_WAIT_SEC)
    assert last_exc is not None
    raise last_exc


def synthesize(client: httpx.Client, utt: Utterance) -> bytes:
    """VOICEVOX の audio_query → synthesis で WAV バイト列を得る。"""
    query = _post_with_retry(
        client, "/audio_query", {"text": utt.text, "speaker": utt.speaker_id}
    ).json()
    query["prePhonemeLength"] = PRE_SILENCE_SEC
    query["postPhonemeLength"] = POST_SILENCE_SEC
    return _post_with_retry(
        client, "/synthesis", {"speaker": utt.speaker_id}, body=query
    ).content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--voicevox-url", default=DEFAULT_VOICEVOX_URL)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.voicevox_url, timeout=TIMEOUT_SEC) as client:
        for utt in UTTERANCES:
            wav = synthesize(client, utt)
            out = out_dir / utt.filename
            out.write_bytes(wav)
            print(f"{out}  ({utt.speaker_name}, {len(wav)} bytes)  {utt.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
