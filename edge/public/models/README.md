# VRM モデル（サンプル）

| 項目 | 内容 |
|---|---|
| ファイル | `VRM1_Constraint_Twist_Sample.vrm`（sha256 `12c2b97e95e700783a6a550dc0eee2d7880aeedccef9ae67bc4c5a2f0f2631a2`） |
| 配布元 | [pixiv/three-vrm](https://github.com/pixiv/three-vrm) の `packages/three-vrm/examples/models/`（dev ブランチ、2026-10-21 取得） |
| 作者・著作権 | pixiv Inc. / (c) 2022 pixiv Inc. |
| ライセンス | VRM パブリック・ライセンス 1.0（<https://vrm.dev/licenses/1.0/>） |

モデル内のライセンス設定（VRMC_vrm.meta）をそのまま転記:

| 項目 | 値 |
|---|---|
| avatarPermission | everyone（誰でもアバター利用＝人格を与えて操作してよい） |
| commercialUsage | corporation（法人の商用利用まで可） |
| allowRedistribution | true |
| modification | allowModificationRedistribution（改変・改変物の再配布可） |
| creditNotation | unnecessary（クレジット表記不要） |
| allowPoliticalOrReligiousUsage | true |
| allowAntisocialOrHateUsage | false |

このサンプルは開発用の既定値（フロントにつながらないときと、テーマのアバターのファイルが無いときに出す）。
本番のアバター（`idochan_office.vrm`・`idochan_festival.vrm`、再配布不可）は git にもビルドにも入れず、
`~/idobata-data/avatars/` に置いてフロントが `/edge/avatars/<name>.vrm` で配信する。どれを出すかは
`server/context/themes.yaml`（WP08。結果は `docs/WP08_result.md`）。
