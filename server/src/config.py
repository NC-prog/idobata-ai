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

    # ローリングコンテキスト（直前の確定転写を system に動的注入）。既定は無効:
    # 実機で、音声が曖昧なときに Qwen3-ASR がコンテキスト文をそのまま出力として返し、
    # 過去の発話がつながった偽の聞き取りが連発した（WP01 追加作業）。コードは残す
    rolling_context_enabled: bool = False
    rolling_context_segments: int = 3
    rolling_context_max_chars: int = 200

    # 保険: 直前の確定転写の焼き直し（コンテキストの吐き出し）を破棄する
    echo_check_segments: int = 3
    # 出力の文字 n-gram のうち、直前の転写にも含まれる割合がこれ以上なら破棄
    echo_overlap_threshold: float = 0.8
    echo_ngram: int = 2
    # これより短い出力（句読点・空白を除く）は判定しない（「はい」の繰り返しを捨てない）
    echo_min_chars: int = 8
    # 出力に拠点の語彙がこの数以上含まれていたら、語彙の一覧の吐き出しとして破棄する
    vocabulary_dump_min_words: int = 6

    # 提案の返事待ち（うん・はい・いらない）は会話の言語に固定して聞き取る（WP04）。
    # 言語判定に任せると「うん」→「嗯」、「はい」→「Hi.」になる。ASR サーバ側に
    # deploy/standalone/asr/chat_template.jinja（force_language 対応）が要る
    force_language_on_reply: bool = True


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
    # 発話開始を検出する前の音声もセグメントに含める長さ（ミリ秒）。Silero は立ち上がりの
    # 後で start を出すので、無いと語頭が欠ける（「うん」→「分」、「国保」→「国宝」。WP04）
    segment_preroll_ms: int = 300
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
    temperature: float = 0.5
    # Qwen3 の思考モードを無効化する（chat_template_kwargs で渡す）
    enable_thinking: bool = False
    # 最後の確定発話からこの秒数で履歴を消して待機に戻る（次の発話は新しい会話）。
    # 環境変数 IDOBATA_IDLE_TIMEOUT_SEC。0 以下で無効
    idle_timeout_sec: float = 90.0
    # ストリーミング応答をこの文字で文に切る（1 文ずつ合成して再生する）。
    # 改行も文の区切り。ASCII の "." は直後が空白のときだけ区切る（英語の応答向け）
    sentence_terminators: str = "。？！?!"
    # 応答の 1 文目に限り、読点でも切る（1 文目が長いと合成待ちで音声開始が遅れるため）。
    # 読点より前がこの文字数未満なら切らない。WP01 では「はい、」だけを先に言わないよう 4 にしたが、
    # WP03 でプロンプトが長くなり、「ええ、最近は…」のような 1 文目を丸ごと合成して音声開始が
    # 1.4 秒を超えたため 2 にした（相槌のあとの間は自然な範囲）
    first_chunk_soft_terminators: str = "、，,"
    first_chunk_min_chars: int = 2
    # 文末記号の直後に続いたら、同じ文に含める閉じ括弧など
    sentence_closers: str = "」』）)】\"'"
    # 日本語の崩れ（英単語の混在・活用の誤り）を抑える。値は WP03 の指示どおり
    presence_penalty: float = 0.3
    # 応答の文に英字の連なり（この文字数以上）があれば、その文だけ日本語で言い直させる
    ascii_run_min: int = 3
    # どの拠点でも英字のまま言ってよい語（拠点 YAML の allow_ascii に足される）
    allow_ascii_common: tuple[str, ...] = ("AED",)
    rephrase_max_tokens: int = 128
    rephrase_prompt: str = (
        "次の文を、意味を変えずに日本語だけで言い直してください。英単語は日本語かカタカナにし、"
        "中国語の漢字（簡体字など）は日本語の漢字にします。"
        "言い直した文だけを出力してください。"
    )
    # 参考資料を使ったときに応答末尾へ付けさせる印（読み上げ前に取り除き、字幕に出典を出す）
    source_marker_example: str = "[出典:1,3]"
    # システムプロンプト（固定事実層 + 参考資料 + 今日）。{...} は persona.py が埋める
    persona_template: str = (
        "あなたは{location_phrase}にいる案内役です。呼び名は「{avatar_name}」です。\n"
        "来訪者の話を急かさず聞きます。敬語ですが硬くありません。\n"
        "【ここで分かっていること】\n"
        "{facts}\n"
        "【話し方の決まり】\n"
        "- 日本語だけで話す。英単語を混ぜない。\n"
        "- 一度に長く話さない。返答は2〜3文にする。相槌や言い換えで受け止め、助言は求められてからにする。\n"
        "- 音声で読み上げるので、箇条書き・記号・絵文字・URL は使わない。\n"
        "- 手続きの要件・期限・持ち物・手順・金額は、【参考資料】にあるときだけ言う。"
        "無いときは「（担当の課）でご案内しますので、お声がけください」と言う。推測で手順を言わない。\n"
        "- 担当の課を案内するときは【ここで分かっていること】の課名と場所を使う。載っていない課の名前を作らない。\n"
        "- 持ち物・必要書類は、【参考資料】に書いてあるものだけを言う。「例えば」で挙げない。"
        "【参考資料】に無いときは、一般的な知識で答えず「持ち物は（担当の課）でご案内しますので、"
        "お声がけください」とだけ言う。\n"
        "- 建物の中の設備（トイレ・エレベーター・授乳室・駐車場など）は、【ここで分かっていること】の"
        "「館内の設備」に載っているものだけ、その場所を言う。載っていない設備の場所を聞かれたら、"
        "場所を作らず「近くの職員にお尋ねください」と言う。課の階は言ってよいが、道順は作らない。\n"
        "- 行事を案内するときは、【参考資料】に書かれた日付（年・月・日。令和の年も見る）が【今日】以降の"
        "ものだけを案内する。過ぎた行事を、これからの行事として言わない。\n"
        "- 行事の日付が今週末・来週末かどうかは【今日】の日付で判断する。それより先の行事は日付だけを言い、"
        "「来週」「来週末」と言わない。\n"
        "- 傘をすすめるのは、【今日】の天気に雨があり、降水確率が30パーセント以上のときだけにする。\n"
        "- 観光地・店・行事・団体の名前は、【参考資料】か【ここで分かっていること】にあるものだけを言う。"
        "無いときは名前を挙げず、担当の課を案内する。【参考資料】の名前は書いてあるとおりに言い、"
        "{city}の外の場所を{city}のものとして挙げない。\n"
        "- 天気・気温は【今日】に書いてあることだけを言う。来訪者の言うことを否定しない。\n"
        "- おすすめを聞かれても、【参考資料】に無い観光地・名所・店の名前は一つも言わない。"
        "そのときは「{city}の観光は商工観光課や観光協会のホームページでご案内しています」のように案内する。\n"
        "- 「参考資料」「ここで分かっていること」という言葉は口に出さない。\n"
        "- 緊急連絡先を案内するときは、名前と電話番号の両方を言う。\n"
        "- 自分がどこにいるかを聞かれたら上の事実で答える。"
        "来訪者に「お住まいの市区町村」を聞かない（ここが{facility_name}である）。\n"
        "- 天気を聞かれたら【今日】の天気の文をそのまま使う。無ければ分からないと言う。\n"
        "- 分からないことは分からないと言う。\n"
        "- 日本語以外で話しかけられたら、日本語で「{foreign_reply}」と短く伝える。\n"
        "- 【参考資料】は来訪者の発話の前に毎回付けて渡す。無いときは「なし」と書いてある。\n"
        "{propose_rules}"
        "【今日】\n"
        "{today}"
    )
    # 毎ターンの来訪者メッセージ（参考資料 + 発話）。参考資料をシステムプロンプトに入れると、
    # 参考資料が変わるたびに後ろの会話履歴まで vLLM のプレフィックスキャッシュが効かなくなるため、
    # 最新の発話の側に付ける（システムプロンプトと履歴はキャッシュされる）
    turn_template: str = "【参考資料】\n{references}\n{source_rule}\n【来訪者の発話】\n{text}"
    # 読み上げ直前の文の検査（言わないことを機械的に止める）。
    # banned_phrases を含む文は読み上げずに捨てる（WARNING）。来訪者に住所を聞き返さない、など
    banned_phrases: tuple[str, ...] = (
        "お住まいの市区町村", "お住まいの自治体", "お住まいの地域の役所", "お住まいの市町村",
    )
    # プロンプトの見出しが漏れたときの言い換え
    # 応答の冒頭で来訪者の言葉をそのまま繰り返す文は読み上げない（8B の癖。一度出ると
    # 履歴に倣って続く）。ただし挨拶を返すのは自然なので残す
    echo_keep: tuple[str, ...] = ("こんにちは", "こんばんは", "おはようございます", "ありがとう")
    leak_replacements: tuple[tuple[str, str], ...] = (
        ("【参考資料】", "手元の資料"), ("参考資料", "手元の資料"),
        ("【ここで分かっていること】", "こちらで分かっていること"),
    )
    # 館内の設備の場所の作り話を止める: 設備の語と場所の語が同じ文にあり、その設備が拠点 YAML の
    # facilities_in_building に無ければ、文を差し替える（WP03 の試験で、AED の「1階ロビー」を
    # トイレ・エレベーターにも当てはめた）
    building_facility_words: tuple[str, ...] = (
        "トイレ", "お手洗い", "化粧室", "多目的トイレ", "エレベーター", "エスカレーター", "授乳", "おむつ",
        "駐車場", "駐輪場", "階段", "喫煙", "自動販売機", "自販機", "コピー機", "公衆電話", "AED",
        "車いす", "車椅子", "休憩", "記載台",
    )
    building_location_pattern: str = r"[0-9０-９一二三四]階|ロビー|奥|隣|そば|横|手前|右|左|突き当た|入口|入り口|正面|向かい|近く"
    unknown_facility_sentence: str = "{name}の場所は、近くの職員にお尋ねください。"
    # 来訪者が登録の無い設備の場所・有無を聞いたら、LLM を通さず unknown_facility_sentence で答える
    facility_question_pattern: str = r"どこ|ありますか|あります？|ある？|場所|行きたい|使いたい|使えますか|探して"
    # 緊急連絡先の名前だけ言って番号を言わなかったときに足す文
    emergency_number_sentence: str = "{name}の番号は、{number}です。"
    # 同意付きの行為を提案してよい状況を、応答の最後のタグで知らせてもらう（WP04）。行為ごとの条件
    propose_rule_head: str = (
        "- 次の状況のときだけ、応答のいちばん最後にタグを 1 つ付ける（タグは読み上げない。"
        "提案の言葉は自分で言わない。こちらで言う）:\n"
    )
    propose_conditions: tuple[tuple[str, str], ...] = (
        ("lookup", ("行事・催し・施設・お知らせ・営業日を知りたがっているのに、【参考資料】に答えが無いとき。"
                    "手続き・相談・困りごとには付けない（それは refer）")),
        ("refer", ("手続き（保険・税・住民票・子育て・介護など）の相談で、担当の課が【ここで分かっていること】から"
                   "分かるとき。または困りごとの打ち明けで相談先があるとき（子育て・介護・生活の困りごとなど）。"
                   "雑談では付けない")),
        ("share", ("行事・季節・畑・魚・天気などの雑談で、来訪者が話したことに個人の事情が含まれず、"
                   "ほかの来訪者に教えても差し支えないとき")),
    )
    # 日本語以外の発話への返事（多言語は site プロファイルで後日）。LLM を通さずこの文を返す
    foreign_reply: str = "申し訳ありません、外国語でのご案内は準備中です。"
    # 参考資料があるターンだけ【話し方の決まり】に足す（無いのに出典の一言を言わせない）
    source_rule: str = (
        "（この資料を使って答えたときは、応答の最後に、使った資料の番号を {marker_example} の形で書く。"
        "「ホームページに載っています」のような出典の一言はこちらで足すので書かない）"
    )
    # 参考資料を使った応答の最後に足す一言（URL は読み上げない）。labels は資料の出どころの言い方
    source_sentence: str = "詳しくは{labels}にも載っています。"
    # 出どころの言い方が無い資料（古いパック）の言い方。org は 市 / 町
    default_source_label: str = "{org}のホームページ"
    # LLM が自分で書いた出典の一言（こちらで足すので捨てる）
    llm_source_sentence_pattern: str = r"^(?:詳しくは|くわしくは).{0,40}(?:ホームページ|サイト|お知らせ).{0,12}載っています"
    no_references_text: str = "なし"


@dataclass(frozen=True)
class KnowledgeConfig:
    """知識パック（kb/build.py の成果物）の検索の設定。"""

    enabled: bool = True
    # 上位何件を参考資料に入れるか、合計の上限字数。指示書の上限は 1,500 字だが、参考資料は毎ターン
    # 変わるのでプレフィックスキャッシュが効かず、1,500 字で初回トークンまで約 0.50 秒、1,000 字で
    # 約 0.34 秒かかった（7560 実測、docs/WP03_result.md）。音声開始 1.3 秒未満を保つため 1,000 字にする
    top_k: int = 5
    max_chars: int = 1000
    # ベクトル検索・BM25（文字 2-gram）それぞれの候補数と、順位の融合（RRF）の定数
    vector_candidates: int = 20
    bm25_candidates: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # 資料を入れる条件（実測で決めた。docs/WP03_result.md の 2-4）:
    # ベクトルの類似度（cos）が min_vector_score 以上、または
    # min_vector_score_with_keyword 以上かつ BM25 で keyword_rank 位以内（語が実際に一致している）
    # Ruri v3 の cos は関係ない発話でも 0.77〜0.84 になるので、ベクトルだけでは切れない
    min_vector_score: float = 0.86
    min_vector_score_with_keyword: float = 0.80
    keyword_rank: int = 5
    # 知識パックの差し替え（manifest.json の更新）を確かめる間隔（秒）
    reload_check_sec: float = 60.0
    # 埋め込みは CPU（GPU の予備枠は YOLO11-pose 用に残す）
    embed_device: str = "cpu"
    embed_threads: int = 4


@dataclass(frozen=True)
class ActionsConfig:
    """相槌と、同意付きの行為（調べる・つなぐ・覚えておく）の設定（WP04）。"""

    # アバターの呼び名（仮。市民公募で決める前提）。環境変数 IDOBATA_AVATAR_NAME
    avatar_name: str = "いどちゃん"
    backchannel_enabled: bool = True
    backchannel_path: Path = SERVER_ROOT / "context" / "backchannel.yaml"
    # 相槌の事前合成にかけてよい時間（秒）。超えたら残りは使うときに合成する
    backchannel_prepare_budget_sec: float = 30.0
    templates_path: Path = SERVER_ROOT / "context" / "actions.yaml"
    # 使う行為（LLM に <propose:…> を出してよいと伝えるもの）
    enabled: tuple[str, ...] = ("lookup", "refer")
    # 返事が分からないときに聞き直す回数（それでも分からなければ辞退扱い）
    max_reask: int = 1
    # LLM がタグを付け忘れたときの規則による補い（8B は付け忘れが多い）。
    # 調べる: 応答が「資料に無い」と言い、発話が行事・営業などを尋ねているとき
    lookup_fallback_reply: str = (
        r"資料に(は)?(ありません|載っていません|見当たりません)|情報が(ありません|見当たりません)|"
        r"手元に(は)?(ありません|情報が)|分かりません|わかりません|把握して(い)?ません"
    )
    lookup_fallback_topic: str = (
        r"行事|イベント|催し|祭り|まつり|やって(い)?ます|営業|開いて(い)?ます|休み|定休|お知らせ|何時まで"
    )
    # 調べるの提案は、発話が尋ねているか行事・営業の話のときだけ受け付ける（雑談に付いたタグは捨てる）
    question_pattern: str = (
        r"[かは？?][。]?$|ますか|ですか|どこ|いつ|何|なに|どう|教えて|知りたい|ある[の？?]|やって"
    )
    # 返事待ちに、返事らしくない長い発話（新しい話）が来たら、聞き直さずに辞退として普通に答える
    reply_max_chars: int = 12
    # つなぐ: 発話が手続き（ReferralConfig.procedure_pattern）の相談を求めているとき
    refer_fallback_cue: str = r"相談|聞きたいこと|困って|分からなくて|わからなくて|どうしたら"


@dataclass(frozen=True)
class LookupConfig:
    """調べる（lookup）の設定。取得先は kb/sources/<site>.yaml の lookup。"""

    sources_dir: Path = REPO_ROOT / "kb" / "sources"
    # 合計でこの秒数で打ち切る
    timeout_sec: float = 5.0
    # 一覧・検索結果の取得の持ち時間（これまでに返ったものだけ使う）と、本文の取得を打ち切る余裕
    list_timeout_sec: float = 2.5
    margin_sec: float = 0.3
    # 一覧・検索結果から読みに行くページ数
    max_pages: int = 3
    max_chars: int = 1000
    # 読んだ本文が見つかったとみなす BM25 の最低点（実測で決める）
    min_score: float = 2.0
    query_max_tokens: int = 32
    query_prompt: str = (
        "来訪者の発話から、市役所のホームページで調べるための検索語を作ってください。"
        "場所・行事・手続き・施設の名前だけを 2〜4 語、空白で区切って出力します。"
        "本人の事情・氏名・体調・家族の話・連絡先は含めません。検索語だけを出力してください。"
    )
    # 一般の検索エンジン（既定 OFF。環境変数 IDOBATA_LOOKUP_WEB_SEARCH=1 で ON）
    web_search_enabled: bool = False
    web_search_url: str = "https://html.duckduckgo.com/html/?q={q}"
    web_search_results: int = 3
    user_agent: str = "idobata-ai-lookup/0.1 (+Fuji Seisakusho LLC; municipal lobby guide pilot)"


@dataclass(frozen=True)
class ReferralConfig:
    """紹介状（本人が同意したものだけを保存する）の設定。"""

    # 保存期間（日）。起動時に古いものを消す。環境変数 IDOBATA_REFERRAL_RETENTION_DAYS
    retention_days: int = 30
    summary_max_tokens: int = 300
    # LLM が lookup を提案しても、発話が手続き・相談ならつなぐ（refer）に置き換える
    # （8B は手続きの相談にも lookup を付けがちなため、規則で補正する）
    procedure_pattern: str = (
        r"手続き?|切り?替え|申請|届け?出|届|証明書|相談|困って|困りごと|聞きたいこと|"
        r"国保|国民健康保険|保険証|年金|介護|児童手当|住民票|戸籍|転入|転出|引っ越し|税金|納付|"
        r"生活保護|障害"
    )
    structure_prompt: str = (
        "以下の来訪者との会話から、担当の課へ伝える紹介状を JSON で作ってください。"
        "department は【課の一覧】のどれか（分からなければ空文字）。"
        "category は 手続き / 相談 / 意見 / その他 のどれか。"
        "summary は来訪者の言葉を 2〜3 文で要約（推測で足さない）。"
        "urgency は 急ぐ / 通常。contact_pref は 窓口で待つ / 電話 / 後日来庁 / 未定。"
        "name と phone は、来訪者が自分で言った場合だけ書き、言っていなければ null。"
        'JSON だけを出力: {"department":"","category":"","summary":"","urgency":"",'
        '"contact_pref":"","name":null,"phone":null}'
    )


@dataclass(frozen=True)
class StaffConfig:
    """職員パネル（/staff/、LAN 内のみ、Basic 認証）の設定。"""

    user: str = "staff"
    # 環境変数 IDOBATA_STAFF_PASSWORD。空ならパネルは使えない
    password: str = ""
    poll_sec: int = 10


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
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    lookup: LookupConfig = field(default_factory=LookupConfig)
    referral: ReferralConfig = field(default_factory=ReferralConfig)
    staff: StaffConfig = field(default_factory=StaffConfig)
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
    edge = EdgeConfig(
        ws_port=int(_parse_float(env, "IDOBATA_EDGE_WS_PORT", EdgeConfig.ws_port)),
        http_port=int(_parse_float(env, "IDOBATA_EDGE_HTTP_PORT", EdgeConfig.http_port)),
        # 職員パネルを LAN から見るときは 0.0.0.0（アクセスは LAN 内のアドレスだけに絞る）
        http_host=env.get("IDOBATA_EDGE_HTTP_HOST", EdgeConfig.http_host).strip() or EdgeConfig.http_host,
    )
    actions = ActionsConfig(
        avatar_name=env.get("IDOBATA_AVATAR_NAME", "").strip() or ActionsConfig.avatar_name)
    lookup = LookupConfig(
        web_search_enabled=env.get("IDOBATA_LOOKUP_WEB_SEARCH", "").strip().lower() in ("1", "true", "yes"))
    referral = ReferralConfig(retention_days=int(_parse_float(
        env, "IDOBATA_REFERRAL_RETENTION_DAYS", ReferralConfig.retention_days)))
    staff = StaffConfig(password=env.get("IDOBATA_STAFF_PASSWORD", "").strip())
    dialogue = DialogueConfig(
        idle_timeout_sec=_parse_float(
            env, "IDOBATA_IDLE_TIMEOUT_SEC", DialogueConfig.idle_timeout_sec
        )
    )

    return AppConfig(
        profile=profile,
        site=site,
        data_dir=data_dir,
        llm=llm,
        audio_input=audio_input,
        audio_output=audio_output,
        dialogue=dialogue,
        edge=edge,
        actions=actions,
        lookup=lookup,
        referral=referral,
        staff=staff,
    )


def _parse_device_id(env: Mapping[str, str], key: str) -> DeviceSpec:
    """音声デバイスの指定を読む。未設定・空なら None（既定デバイス）。"""
    return parse_device(env.get(key, ""))


def _parse_float(env: Mapping[str, str], key: str, default: float) -> float:
    """秒数などの数値を読む。未設定・空なら default。"""
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} は数値で指定してください: {raw!r}") from exc


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
