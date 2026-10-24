"""「役」の骨組み（WP11 C-1）。

案内役は役の一つ（総合受付 reception）で、今回はお知らせを一つ読む広報（publicity）を足した。
進行役（オーケストレータ）は main.py の状態機械の中にあり、役の選択は ``select_role`` の 1 か所に閉じ込める。
今後の役（ロビーの相談員・井戸端ファシリテーター）は Role を足して ``select_role`` に条件を足すだけにする。

- prompt_block: システムプロンプトの【話し方の決まり】の後ろに足す段落（reception は人物像の段落。文言は WP05 のまま）
- actions: その役で提案できる行為
- layout: エッジの構図（normal / news）
"""

from __future__ import annotations

from dataclasses import dataclass

from .persona import Persona

RECEPTION = "reception"
PUBLICITY = "publicity"

# 広報の決まり（お知らせは夜に作って安全装置を通した要約を読むだけなので、LLM には渡らない。役の説明として持つ）
PUBLICITY_BLOCK = (
    "【広報の決まり】\n"
    "- お知らせを一つだけ読む係。読むのは職員と夜の安全装置を通った要約だけで、書き足さない。\n"
    "- 読み終えたら、QR コードからページを見られることを伝えて、総合受付に戻る。\n"
)


@dataclass(frozen=True)
class Role:
    """役（WP11 C-1）。"""

    name: str
    prompt_block: str
    actions: tuple[str, ...]
    layout: str


@dataclass
class RoleState:
    """役を選ぶ材料（進行役が持つ）。今は「お知らせを紹介している最中か」だけ。"""

    presenting_news: bool = False


def reception_role(persona: Persona | None, facility_name: str, actions: tuple[str, ...]) -> Role:
    """総合受付（今までの案内役）。prompt_block は人物像の段落（WP05 の persona.block をそのまま）。"""
    return Role(RECEPTION, persona.block(facility_name) if persona else "", actions, "normal")


def publicity_role() -> Role:
    """広報（お知らせを一つ読む係）。"""
    return Role(PUBLICITY, PUBLICITY_BLOCK, (), "news")


class Roles:
    """拠点で使う役の一覧と、いまの役。"""

    def __init__(self, reception: Role, publicity: Role) -> None:
        self.reception = reception
        self.publicity = publicity
        self.state = RoleState()
        self.current = reception

    def select(self) -> tuple[Role, bool]:
        """いまの状態で役を選び直す。(役, 変わったか)。"""
        role = select_role(self.state, self)
        changed = role is not self.current
        self.current = role
        return role, changed


def select_role(state: RoleState, roles: Roles) -> Role:
    """役の選択（ここだけで決める）。今回は reception → publicity → reception の 1 往復だけ。"""
    if state.presenting_news:
        return roles.publicity
    return roles.reception
