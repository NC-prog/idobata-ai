"""全設定の集約モジュール。

アプリケーション全体の設定を dataclass で定義し、モジュールレベルの
`config` インスタンスとして公開する。

実行プロファイルは環境変数 ``IDOBATA_PROFILE`` で切り替える:

- ``standalone``: ASR・LLM・TTS すべて 127.0.0.1（7560 単体デモ）
- ``site``: ASR・TTS は 127.0.0.1、LLM は ``IDOBATA_BACKEND_URL``（Tailscale 経由の T2／3660）

環境変数は ``deploy/standalone/.env``（または ``IDOBATA_ENV_FILE`` で指定したファイル）
からも読み込む。既にプロセス環境にある値が優先される。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# リポジトリルート（server/src/config.py から 2 つ上）
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SERVER_ROOT: Path = REPO_ROOT / "server"
DEFAULT_ENV_FILE: Path = REPO_ROOT / "deploy" / "standalone" / ".env"

LOCALHOST = "127.0.0.1"
ASR_PORT = 8005
LLM_PORT = 8001
VOICEVOX_PORT = 50021


# 音声デバイスの指定: sounddevice の番号、名前の一部、None（既定デバイス）
DeviceSpec = int | str | None


class Profile(str, Enum):
    """実行プロファイル。"""

    STANDALONE = "standalone"
    SITE = "site"


class ConfigError(RuntimeError):
    """設定不備（起動時に検出して停止する）。"""


def load_env_file(path: Path, environ: dict[str, str] | None = None) -> None:
    """KEY=VALUE 形式の .env を読み、未設定のキーだけ環境変数に入れる。

    ``${HOME}`` のような参照は ``os.path.expandvars`` で展開する。
    ファイルが無ければ何もしない。
    """
    env = os.environ if environ is None else environ
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = os.path.expandvars(value)


@dataclass(frozen=True)
class LLMConfig:
    """vLLM（OpenAI 互換）接続の設定。"""

    gateway_url: str = f"http://{LOCALHOST}:{LLM_PORT}/v1"

    # モデル名 → 直接接続 URL（Gateway 障害時のフォールバック）。
    # 井戸端AIでは Gateway を置かないため既定は空。
    direct_urls: dict[str, str] = field(default_factory=dict)

    # 用途別モデル名
    chat_model: str = "/models/llm"
    reasoning_model: str = "/models/llm"

    api_key: str = "dummy"
    timeout: float = 60.0
    reasoning_timeout: float = 180.0
    max_retries: int = 3
    max_tokens: int = 512
    reasoning_max_tokens: int = 2048


@dataclass(frozen=True)
class ASRConfig:
    """Qwen3-ASR サービスの設定。"""

    base_url: str = f"http://{LOCALHOST}:{ASR_PORT}/v1"
    model: str = "/models/asr"
    temperature: float = 0.0
    timeout: float = 60.0

    # コンテキストバイアシング（拠点の地名・施設名・課名リスト）
    context_biasing: bool = True
    site_context_dir: Path = SERVER_ROOT / "context"

    # ローリングコンテキスト（直前の確定転写を動的注入）
    rolling_context: bool = True
    rolling_context_segments: int = 3
    rolling_context_max_chars: int = 200


@dataclass(frozen=True)
class VoicevoxConfig:
    """VOICEVOX TTS の設定（tts_client.py 互換）。"""

    base_url: str = f"http://{LOCALHOST}:{VOICEVOX_PORT}"
    speaker_id: int = 2
    speed_scale: float = 1.0
    pitch_scale: float = 0.0
    intonation_scale: float = 1.0
    volume_scale: float = 1.0
    # 文末の無音（秒）。文単位で続けて再生するので、文と文の間はこれ（+ 次文の前無音）で作る
    post_phoneme_length: float = 0.1
    timeout: float = 30.0
    # 接続失敗・5xx の再試行回数と待ち時間（秒）
    max_retries: int = 2
    retry_wait_sec: float = 0.3

    # ユーザー辞書（拠点 YAML の読み付きの語を起動時に登録する）
    user_dict_enabled: bool = True
    user_dict_word_type: str = "PROPER_NOUN"
    # 優先度 0〜10。8 以下だとシステム辞書の読み（能美→ノミ）に負けた（VOICEVOX 0.25.2 で確認）
    user_dict_priority: int = 9
    # YAML で accent を省略した語のアクセント核の位置
    user_dict_default_accent: int = 1


@dataclass(frozen=True)
class VADConfig:
    """Silero VAD の設定。"""

    backend: str = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    pre_roll_ms: int = 30
    max_segment_sec: float = 7.0
    sample_rate: int = 16000
    frame_size: int = 512

    # ソフト分割: max_segment 到達前にエネルギーの谷で自然に分割する
    soft_split_enabled: bool = True
    soft_split_lookback_ms: int = 1500
    soft_split_rise_threshold: float = 0.003


@dataclass(frozen=True)
class AudioInputConfig:
    """音声入力の設定。"""

    sample_rate: int = 16000
    channels: int = 1
    frame_size: int = 512
    device_id: DeviceSpec = None
    wav_speed: float = 1.0


@dataclass(frozen=True)
class AudioOutputConfig:
    """音声出力（TTS 再生先）の設定。"""

    device_id: DeviceSpec = None
    # 再生タイムアウト = WAV の実時間 + この秒数。超えたらストリームを強制的に閉じる
    play_timeout_margin_sec: float = 2.0
    # OutputStream.write() 1 回あたりの長さ（秒）。中断要求はこの粒度で確認する
    write_block_sec: float = 0.1
    # 連続再生で次の文の合成を待つ間に書く無音の長さ（秒）。次の文が来たらこの粒度で繋がる
    idle_fill_sec: float = 0.02


@dataclass(frozen=True)
class SpeechGateConfig:
    """自己発話ゲートの設定。"""

    # TTS 再生終了後、VAD を再開するまでの後マージン（秒）
    post_margin_sec: float = 0.7
    # 安全装置: mute(expected_sec) から「想定再生時間 + 後マージン + この秒数」で自動 unmute する
    # （再生処理が戻らなくてもゲートが開くように）
    safety_margin_sec: float = 2.0


@dataclass(frozen=True)
class DialogueConfig:
    """最小対話（dialogue.py）の設定。"""

    # 履歴として LLM に渡す直近ターン数（1 ターン = ユーザー発話 + 応答）
    history_turns: int = 6
    max_tokens: int = 256
    temperature: float = 0.7
    # Qwen3 の思考モードを無効化する（chat_template_kwargs で渡す）
    enable_thinking: bool = False
    # ストリーミング応答をこの文字で文に切る（1 文ずつ合成して再生する）。
    # 改行も文の区切り。ASCII の "." は直後が空白のときだけ区切る（英語の応答向け）
    sentence_terminators: str = "。？！?!"
    # 応答の 1 文目に限り、読点でも切る（1 文目が長いと合成待ちで音声開始が遅れるため）。
    # 読点より前がこの文字数未満なら切らない（「はい、」だけを先に言わない）
    first_chunk_soft_terminators: str = "、，,"
    first_chunk_min_chars: int = 4
    # 文末記号の直後に続いたら、同じ文に含める閉じ括弧など
    sentence_closers: str = "」』）)】\"'"
    system_prompt: str = (
        "あなたは「井戸端AI」です。市役所のロビーにいる、話を急かさない聞き役です。\n"
        "- 敬語ですが、硬くなりすぎず、やわらかく話します。\n"
        "- 一度に長く話しません。返答は2〜3文にします。\n"
        "- 相手の話をよく聞き、相槌や言い換えで受け止めます。助言は求められてからにします。\n"
        "- 手続きの詳細（必要書類・期限・金額など）は断定せず、担当窓口での確認をすすめます。\n"
        "- 音声で読み上げるので、箇条書き・記号・絵文字・URL は使いません。\n"
        "- 相手が日本語以外で話したら、その言語で短く答えます。"
    )


@dataclass(frozen=True)
class CaptionConfig:
    """字幕表示の設定。"""

    max_line_chars: int = 35


@dataclass(frozen=True)
class EdgeConfig:
    """エッジ（ブラウザのアバター画面）との接続の設定。"""

    ws_host: str = LOCALHOST
    ws_port: int = 8765
    # エッジページの静的配信（edge/ を Vite でビルドした成果物）
    http_host: str = LOCALHOST
    http_port: int = 8080
    static_dir: Path = SERVER_ROOT / "static" / "edge"
    # playback_ended を待つ上限 = 再生終了の見込み + この秒数
    playback_timeout_margin_sec: float = 3.0
    # 季節（背景 edge/public/backgrounds/<site>_<season>.jpg）。3〜5 月が spring、以降 3 か月ずつ
    seasons: tuple[str, str, str, str] = ("spring", "summer", "autumn", "winter")


@dataclass(frozen=True)
class AppConfig:
    """アプリケーション全体の設定。"""

    profile: Profile = Profile.STANDALONE
    site: str = "etajima"
    data_dir: Path = Path.home() / "idobata-data"
    llm: LLMConfig = field(default_factory=LLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    voicevox: VoicevoxConfig = field(default_factory=VoicevoxConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    audio_input: AudioInputConfig = field(default_factory=AudioInputConfig)
    audio_output: AudioOutputConfig = field(default_factory=AudioOutputConfig)
    speech_gate: SpeechGateConfig = field(default_factory=SpeechGateConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)


def build_config(environ: Mapping[str, str] | None = None) -> AppConfig:
    """環境変数からプロファイルを解決して AppConfig を組み立てる。

    Raises:
        ConfigError: 未知のプロファイル、または site で IDOBATA_BACKEND_URL が未設定。
    """
    env = os.environ if environ is None else environ

    raw_profile = env.get("IDOBATA_PROFILE", Profile.STANDALONE.value).strip().lower()
    try:
        profile = Profile(raw_profile)
    except ValueError as exc:
        raise ConfigError(
            f"IDOBATA_PROFILE が不正です: {raw_profile!r}（standalone / site）"
        ) from exc

    site = env.get("IDOBATA_SITE", "etajima").strip() or "etajima"
    data_dir = Path(env.get("IDOBATA_DATA_DIR", str(Path.home() / "idobata-data")))

    if profile is Profile.SITE:
        backend = env.get("IDOBATA_BACKEND_URL", "").strip()
        if not backend:
            raise ConfigError(
                "IDOBATA_PROFILE=site では IDOBATA_BACKEND_URL（LLM の /v1 URL）が必須です"
            )
        llm = LLMConfig(gateway_url=backend.rstrip("/"))
    else:
        llm = LLMConfig()

    audio_input = AudioInputConfig(
        device_id=_parse_device_id(env, "IDOBATA_AUDIO_INPUT_DEVICE")
    )
    audio_output = AudioOutputConfig(
        device_id=_parse_device_id(env, "IDOBATA_AUDIO_OUTPUT_DEVICE")
    )

    return AppConfig(
        profile=profile,
        site=site,
        data_dir=data_dir,
        llm=llm,
        audio_input=audio_input,
        audio_output=audio_output,
    )


def _parse_device_id(env: Mapping[str, str], key: str) -> DeviceSpec:
    """音声デバイスの指定を読む。未設定・空なら None（既定デバイス）。"""
    return parse_device(env.get(key, ""))


def parse_device(raw: str | None) -> DeviceSpec:
    """音声デバイスの指定（番号、または名前の一部）を sounddevice に渡せる形にする。

    番号は USB 機器の抜き差しでずれるので、名前（例: ``hw:0,0``、
    ``alsa_output.pci-0000_00_1f.3.analog-stereo``）でも指定できる。
    名前は sounddevice が入力・出力それぞれの機器名の部分一致で解決する。
    """
    text = (raw or "").strip()
    if not text:
        return None
    return int(text) if text.isdigit() else text


def _load_default() -> AppConfig:
    """モジュール import 時の既定設定。

    .env を読み込んだうえで組み立てる。site プロファイルの不備は
    ここでは standalone に落とさず ConfigError を送出する（起動時に止める）。
    """
    env_file = Path(os.environ.get("IDOBATA_ENV_FILE", str(DEFAULT_ENV_FILE)))
    load_env_file(env_file)
    return build_config()


# モジュールレベルのシングルトンインスタンス
config = _load_default()
