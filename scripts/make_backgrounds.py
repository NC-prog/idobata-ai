"""背景画像（イラスト調）を作る。

- 江田島: 海と島影と牡蠣筏。季節は光の色味だけ変える（季節感が薄い土地なので）。
- 安芸太田: 重なる山並み。秋は紅葉の暖色、冬は雪の白、春は若葉、夏は深い緑。
- アバターは画面中央に立つので、景色は下半分と左右に寄せ、中央上部は空（単純な面）にする。
- 白いシャツが溶けないよう、下ほど濃い。

出力: edge/public/backgrounds/<site>_<season>.jpg（1920×1080。縦置きは中央をトリミングして使う）
実写に差し替えるときは ~/idobata-data/backgrounds/ に同じ名前で置けば、そちらが優先される。

    cd server && uv run python ../scripts/make_backgrounds.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "edge" / "public" / "backgrounds"
W, H = 1920, 1080


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def vgrad(top: tuple[int, int, int], bottom: tuple[int, int, int], h: int = H) -> Image.Image:
    img = Image.new("RGB", (W, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        d.line([(0, y), (W, y)], fill=lerp(top, bottom, y / max(1, h - 1)))
    return img


def ridge(rng: random.Random, base_y: int, amp: int, freq: float, phase: float) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for x in range(0, W + 40, 20):
        y = base_y
        y += amp * math.sin(x * freq + phase)
        y += amp * 0.45 * math.sin(x * freq * 2.3 + phase * 1.7)
        y += amp * 0.2 * math.sin(x * freq * 5.1 + phase * 0.6)
        y += rng.uniform(-amp * 0.05, amp * 0.05)
        pts.append((x, int(y)))
    return pts


def draw_mountains(img: Image.Image, layers: list[tuple[int, int, tuple[int, int, int]]], seed: int) -> None:
    """layers: (base_y, amp, color) を奥から手前へ。"""
    rng = random.Random(seed)
    for i, (base_y, amp, color) in enumerate(layers):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pts = ridge(rng, base_y, amp, 0.0022 + i * 0.0007, rng.uniform(0, 6))
        poly = [(0, H)] + pts + [(W, H)]
        d.polygon(poly, fill=color + (255,))
        # 奥の層ほど霞ませる
        blur = max(0, (len(layers) - 1 - i)) * 1.5
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        img.paste(layer, (0, 0), layer)


def add_mist(img: Image.Image, y: int, strength: int, color: tuple[int, int, int]) -> None:
    mist = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(mist)
    for k in range(60):
        a = int(strength * (1 - k / 60))
        d.rectangle([0, y + k * 3, W, y + k * 3 + 3], fill=color + (a,))
    mist = mist.filter(ImageFilter.GaussianBlur(18))
    img.paste(mist, (0, 0), mist)


def vignette(img: Image.Image, dark: float = 0.18) -> Image.Image:
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for y in range(H):
        v = int(255 * dark * (y / H) ** 1.6)
        d.line([(0, y), (W, y)], fill=v)
    black = Image.new("RGB", (W, H), (10, 14, 22))
    return Image.composite(black, img, mask)


def akiota(season: str) -> Image.Image:
    if season == "autumn":
        sky_t, sky_b = (236, 232, 222), (246, 226, 196)
        layers = [(520, 60, (214, 170, 150)), (600, 70, (196, 120, 90)), (690, 80, (176, 86, 58)),
                  (790, 90, (150, 62, 40)), (900, 70, (104, 48, 34))]
        mist_c = (250, 236, 214)
    elif season == "winter":
        sky_t, sky_b = (222, 228, 236), (236, 238, 240)
        layers = [(520, 60, (206, 214, 222)), (600, 70, (184, 194, 206)), (690, 80, (160, 172, 188)),
                  (790, 90, (132, 146, 164)), (900, 70, (96, 108, 126))]
        mist_c = (244, 246, 250)
    elif season == "spring":
        sky_t, sky_b = (226, 236, 240), (240, 236, 226)
        layers = [(520, 60, (190, 206, 190)), (600, 70, (160, 188, 150)), (690, 80, (124, 160, 112)),
                  (790, 90, (92, 130, 86)), (900, 70, (64, 96, 66))]
        mist_c = (246, 240, 236)
    else:  # summer
        sky_t, sky_b = (190, 214, 232), (226, 236, 240)
        layers = [(520, 60, (150, 180, 170)), (600, 70, (104, 146, 122)), (690, 80, (72, 118, 92)),
                  (790, 90, (48, 92, 70)), (900, 70, (30, 64, 50))]
        mist_c = (236, 244, 244)
    img = vgrad(sky_t, sky_b)
    draw_mountains(img, layers, seed=hash(season) & 0xFFFF)
    add_mist(img, 560, 110, mist_c)
    add_mist(img, 700, 70, mist_c)
    if season == "winter":
        # 粉雪
        rng = random.Random(7)
        d = ImageDraw.Draw(img, "RGBA")
        for _ in range(500):
            x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1, 3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 120))
    return vignette(img, 0.22)


def etajima(season: str) -> Image.Image:
    # 季節は光の色味だけ
    tint = {
        "spring": ((222, 232, 236), (236, 230, 226), (130, 180, 194), (78, 132, 152)),
        "summer": ((176, 212, 236), (222, 236, 244), (84, 160, 192), (46, 112, 144)),
        "autumn": ((232, 228, 214), (240, 224, 200), (122, 170, 180), (74, 122, 138)),
        "winter": ((214, 222, 232), (230, 232, 236), (114, 150, 170), (70, 104, 126)),
    }[season]
    sky_t, sky_b, sea_t, sea_b = tint
    horizon = 640
    img = vgrad(sky_t, sky_b, horizon)
    full = Image.new("RGB", (W, H))
    full.paste(img, (0, 0))
    sea = vgrad(sea_t, sea_b, H - horizon)
    full.paste(sea, (0, horizon))
    d = ImageDraw.Draw(full, "RGBA")
    # 島影（左右に寄せる。中央はアバター）
    rng = random.Random(3)
    island_c = lerp(sea_b, (40, 60, 70), 0.35)
    for cx, w, h in [(260, 520, 90), (1700, 460, 70), (-60, 400, 60), (2000, 380, 55)]:
        pts = []
        for i in range(0, w + 1, 16):
            x = cx - w // 2 + i
            t = i / w
            y = horizon - h * math.sin(math.pi * t) * (0.75 + 0.25 * math.sin(i * 0.05 + cx))
            pts.append((x, int(y)))
        d.polygon([(cx - w // 2, horizon)] + pts + [(cx + w // 2, horizon)], fill=island_c + (255,))
    # 牡蠣筏（横長の細い板を奥に並べる）
    raft_c = lerp(sea_b, (30, 40, 44), 0.5)
    for row, (y, scale) in enumerate([(700, 0.6), (760, 0.8), (840, 1.0)]):
        for k in range(-1, 9):
            x = 120 + k * 230 + (row % 2) * 90
            if 760 < x < 1160:  # 中央は空ける
                continue
            w, h = int(110 * scale), int(8 * scale)
            d.rectangle([x, y, x + w, y + h], fill=raft_c + (230,))
            for j in range(0, w, 14):
                d.line([(x + j, y), (x + j, y + h)], fill=lerp(raft_c, (90, 100, 104), 0.5) + (200,), width=1)
    # 水面のきらめき
    for _ in range(900):
        x, y = rng.uniform(0, W), rng.uniform(horizon + 10, H)
        if 700 < x < 1220 and y < 900:
            continue
        ln = rng.uniform(6, 30)
        a = int(rng.uniform(20, 70))
        d.line([(x, y), (x + ln, y)], fill=(255, 255, 255, a), width=1)
    # 水平線を少しぼかす
    band = full.crop((0, horizon - 6, W, horizon + 6)).filter(ImageFilter.GaussianBlur(3))
    full.paste(band, (0, horizon - 6))
    return vignette(full, 0.14)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for season in ("spring", "summer", "autumn", "winter"):
        akiota(season).save(OUT / f"akiota_{season}.jpg", quality=88)
        etajima(season).save(OUT / f"etajima_{season}.jpg", quality=88)
    print(f"wrote 8 images to {OUT}")


if __name__ == "__main__":
    main()
