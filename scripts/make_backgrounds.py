"""エッジの背景の仮置き（拠点の色味の単色グラデーション、季節で 4 種）を作る（WP08）。

江田島は海の青緑、安芸太田は山の深緑。季節で少しずつ色を振る。アバターの白いシャツが溶けないよう、
下ほど濃くする。実写に差し替えるときは ~/idobata-data/backgrounds/ に同じ名前で置けば、そちらが優先される。

    cd server && uv run python ../scripts/make_backgrounds.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "edge" / "public" / "backgrounds"
SIZE = 1920  # 正方形（縦置き・横置きのどちらでも cover で切り出す）

# (上の色, 下の色) を RGB で。上は頭の後ろ、下は胸のあたり（白いシャツが乗る）
COLORS: dict[str, dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "etajima": {  # 海の青緑
        "spring": ((112, 160, 160), (34, 82, 88)),
        "summer": ((84, 160, 170), (18, 74, 90)),
        "autumn": ((118, 146, 140), (40, 70, 72)),
        "winter": ((106, 134, 146), (30, 54, 66)),
    },
    "akiota": {  # 山の深緑
        "spring": ((128, 158, 118), (40, 74, 46)),
        "summer": ((98, 146, 98), (24, 66, 36)),
        "autumn": ((140, 142, 104), (54, 64, 38)),
        "winter": ((112, 128, 116), (34, 50, 42)),
    },
}


def gradient(top: tuple[int, int, int], bottom: tuple[int, int, int], seed: int) -> np.ndarray:
    """縦のグラデーションに、頭の後ろを少し明るくする楕円と、縞が出ないための微かな粒を足す（BGR、uint8）。"""
    y = np.linspace(0.0, 1.0, SIZE)[:, None, None]
    t = y**1.3
    img = (1 - t) * np.array(top, float) + t * np.array(bottom, float)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    glow = np.exp(-(((xx - 0.5) / 0.32) ** 2 + ((yy - 0.3) / 0.26) ** 2))[..., None]
    img = img * (1 + 0.12 * glow)
    img = img + np.random.default_rng(seed).normal(0, 1.2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)[..., ::-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (site, seasons) in enumerate(COLORS.items()):
        for j, (season, (top, bottom)) in enumerate(seasons.items()):
            path = OUT / f"{site}_{season}.jpg"
            cv2.imwrite(str(path), gradient(top, bottom, i * 10 + j), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(path)


if __name__ == "__main__":
    main()
