"""テンプレートベースのアナウンス生成。

friendly / formal の2スタイルで、シーンごとに2-3パターンの
テンプレートを持ち、ランダムに選択して自然なMCを実現する。

generate() は表示用（name を使用）、generate_for_tts() は
読み上げ用（reading を使用）のテキストを返す。

TTS 読み上げ規則:
- 英語表現はカタカナ連結に置換（間延び防止）
- 姓名の reading はスペース除去して敬称まで一続きにする
- 意図的な間は読点「、」で明示する
"""

from __future__ import annotations

import random
import re
from enum import Enum
from typing import Any

from .timeofday import greeting_for


class Scene(Enum):
    """アナウンスシーンの定義。"""

    OPENING = "opening"
    PRESENTER_INTRO = "presenter_intro"
    THANK_YOU = "thank_you"
    THANK_YOU_QA = "thank_you_qa"
    SUMMARIZING = "summarizing"
    SUMMARY_INTRO = "summary_intro"
    NEXT_PRESENTER = "next_presenter"
    CLOSING = "closing"
    PART_CLOSING = "part_closing"


# テンプレート辞書: style → scene → テンプレートリスト
# {name}: 表示用名前、{reading}: 読み上げ用、{dept}: 部署、{topic}: 発表題目
# {title}: 会議タイトル、{greeting}: 時間帯の挨拶（src/timeofday.py）
_TEMPLATES: dict[str, dict[Scene, list[str]]] = {
    "friendly": {
        Scene.OPENING: [
            "皆さん、{greeting}！{title}を始めましょう！",
            "お待たせしました！{title}のスタートです！",
            "それでは、{title}を開会いたします！本日もよろしくお願いします！",
        ],
        Scene.PRESENTER_INTRO: [
            "それでは次の発表です！{dept}の{name}さん、お願いします！",
            "続きまして、{dept}の{name}さんから「{topic}」についてお話しいただきます！",
            "お待たせしました！{dept}の{name}さん、「{topic}」よろしくお願いします！",
        ],
        Scene.THANK_YOU: [
            "{name}さん、ありがとうございました！",
            "{name}さん、素晴らしい発表をありがとうございました！",
        ],
        Scene.THANK_YOU_QA: [
            "{name}さん、発表ありがとうございました！この発表について、質問やコメントがあればお願いします！",
            "{name}さん、ありがとうございました！質問やコメントがあれば、ぜひお願いします！",
        ],
        Scene.SUMMARIZING: [
            "発表内容をただいま要約していますので、しばらくお待ちください！",
            "要約を準備中です。少々お待ちくださいね！",
            "ただいま発表のまとめを作成中です！もう少しお待ちください！",
        ],
        Scene.SUMMARY_INTRO: [
            "それでは、{name}さんの発表のまとめです！",
            "{name}さんの発表を振り返ってみましょう！",
        ],
        Scene.NEXT_PRESENTER: [
            "次の発表に移りましょう！",
            "それでは、次の発表者をお呼びしますね！",
        ],
        Scene.CLOSING: [
            "以上で{title}を終了します！皆さん、お疲れさまでした！",
            "{title}、これにて閉会です！本日はありがとうございました！",
            "皆さん、本日の{title}はここまでです！ご参加ありがとうございました！",
        ],
        Scene.PART_CLOSING: [
            "これでPlatform Serviceの発表を終わります。ありがとうございました！",
            "以上でPlatform Serviceのパートは終了です！ありがとうございました！",
        ],
    },
    "formal": {
        Scene.OPENING: [
            "ただいまより{title}を開会いたします。",
            "{title}を開始いたします。本日はご参加いただきありがとうございます。",
            "定刻になりましたので、{title}を始めさせていただきます。",
        ],
        Scene.PRESENTER_INTRO: [
            "続きまして、{dept}の{name}様より「{topic}」についてご発表いただきます。",
            "それでは、{dept}の{name}様、「{topic}」のご発表をお願いいたします。",
            "次に、{dept}の{name}様から「{topic}」についてお話しいただきます。",
        ],
        Scene.THANK_YOU: [
            "{name}様、ご発表ありがとうございました。",
            "{name}様、ありがとうございました。",
        ],
        Scene.THANK_YOU_QA: [
            "{name}様、ご発表ありがとうございました。この発表についてご質問やコメントがございましたらお願いいたします。",
            "{name}様、ありがとうございました。ご質問やコメントがございましたら、お願いいたします。",
        ],
        Scene.SUMMARIZING: [
            "ただいま発表内容の要約を作成しております。しばらくお待ちください。",
            "要約を準備しておりますので、少々お待ちいただけますでしょうか。",
            "発表のまとめを作成中でございます。今しばらくお待ちください。",
        ],
        Scene.SUMMARY_INTRO: [
            "{name}様のご発表のまとめをご覧ください。",
            "続いて、{name}様のご発表の要約をお示しいたします。",
        ],
        Scene.NEXT_PRESENTER: [
            "それでは、次のご発表に移ります。",
            "続きまして、次の発表者をご紹介いたします。",
        ],
        Scene.CLOSING: [
            "以上をもちまして{title}を閉会いたします。ありがとうございました。",
            "{title}を終了いたします。本日はご参加いただきありがとうございました。",
            "これにて{title}を閉会とさせていただきます。お疲れさまでした。",
        ],
        Scene.PART_CLOSING: [
            "これでPlatform Serviceの発表を終わります。ありがとうございました。",
            "以上でPlatform Serviceのパートは終了でございます。ありがとうございました。",
        ],
    },
}


# ============================================================
# TTS 読み上げ用テキスト変換
# ============================================================

# 英語表現 → カタカナ連結の置換辞書
# VOICEVOX は英単語間にスペースがあると不自然な間が入るため、
# 一息で読む単位にカタカナ連結する
_TTS_REPLACEMENTS: dict[str, str] = {
    "「Who am I?」": "フーアムアイ",
    "Who am I?": "フーアムアイ",
    "Platform Service": "プラットフォームサービス",
}

# CJK 文字クラス（ひらがな・カタカナ・漢字・CJK記号）
_CJK_CHAR = r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uf900-\ufaff]"
_CJK_SPACE_RE = re.compile(rf"({_CJK_CHAR})\s+({_CJK_CHAR})")


def apply_tts_replacements(text: str) -> str:
    """TTS 読み上げ用のテキスト変換を適用する。

    1. 英語表現をカタカナ連結に置換
    2. CJK 文字間の不要なスペースを除去

    Args:
        text: 変換対象テキスト

    Returns:
        TTS 用に変換済みのテキスト
    """
    for eng, kana in _TTS_REPLACEMENTS.items():
        text = text.replace(eng, kana)
    # CJK 文字間のスペースを除去（「プラットフォームサービス 発表」→ 連結）
    # 正規表現は1回の置換で隣接ペアしか処理しないため繰り返す
    prev = ""
    while prev != text:
        prev = text
        text = _CJK_SPACE_RE.sub(r"\1\2", text)
    return text


def generate(scene: Scene, style: str = "friendly", **kwargs: Any) -> str:
    """表示用アナウンステキストを生成する。

    テンプレートの {name} は表示用名前がそのまま埋め込まれる。

    Args:
        scene: アナウンスシーン
        style: 'friendly' または 'formal'
        **kwargs: テンプレート変数（name, dept, topic, title 等）

    Returns:
        展開済みアナウンステキスト
    """
    templates = _TEMPLATES[style][scene]
    template = random.choice(templates)
    kwargs.setdefault("greeting", greeting_for())  # 切り出しの挨拶も応答と同じ時刻判定
    return template.format(**kwargs)


def generate_for_tts(scene: Scene, style: str = "friendly", **kwargs: Any) -> str:
    """TTS 読み上げ用アナウンステキストを生成する。

    - {name} を {reading}（スペース除去済み）に差し替える
    - 英語表現をカタカナ連結に置換する
    - CJK 文字間の不要なスペースを除去する

    Args:
        scene: アナウンスシーン
        style: 'friendly' または 'formal'
        **kwargs: テンプレート変数（name, reading, dept, topic, title 等）

    Returns:
        TTS 用に変換済みのアナウンステキスト
    """
    reading = kwargs.pop("reading", None)
    if reading is not None:
        # 姓名間のスペースを除去して一息で読めるようにする
        # 例: "すずき いちろう" → "すずきいちろう"
        # テンプレート側の敬称（さん/様）と連結され
        # "すずきいちろうさん" と自然に読み上げられる
        kwargs["name"] = reading.replace(" ", "").replace("\u3000", "")
    # dept_reading が指定されていれば TTS 用に dept を差し替える
    dept_reading = kwargs.pop("dept_reading", None)
    if dept_reading is not None:
        kwargs["dept"] = dept_reading
    text = generate(scene, style, **kwargs)
    return apply_tts_replacements(text)
