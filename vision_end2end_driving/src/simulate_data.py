import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def draw_dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill, width: int) -> None:
    for p0, p1 in zip(points[:-1], points[1:]):
        if random.random() > 0.45:
            draw.line([p0, p1], fill=fill, width=width)


def lane_points(width: int, height: int, center_shift: float, curvature: float, lane_half_width: float):
    ys = np.linspace(height * 0.40, height + 8, 24)
    progress = (ys - height * 0.40) / (height * 0.60)
    center = width / 2 + center_shift * width * 0.28 + curvature * width * (progress ** 2 - 0.25)
    lane_width = lane_half_width * (0.24 + 0.76 * progress)
    left = list(zip(center - lane_width, ys))
    right = list(zip(center + lane_width, ys))
    return left, right, center


def generate_frame(width: int, height: int, idx: int, split_hint: str) -> tuple[Image.Image, dict]:
    curvature = float(np.clip(np.random.normal(0, 0.42), -1, 1))
    lateral_offset = float(np.clip(np.random.normal(0, 0.33), -1, 1))
    traffic_density = float(np.random.beta(1.2, 5.5))
    obstacle = random.random() < 0.12
    rain = random.random() < 0.15
    fog = random.random() < 0.12
    glare = random.random() < 0.08
    bad_frame = random.random() < 0.015

    sky = (95 + random.randint(-12, 24), 134 + random.randint(-10, 22), 164 + random.randint(-8, 20))
    img = Image.new("RGB", (width, height), sky)
    draw = ImageDraw.Draw(img, "RGBA")

    horizon = int(height * random.uniform(0.36, 0.46))
    draw.rectangle([0, horizon, width, height], fill=(42, 70, 55, 255))

    road_top = width * random.uniform(0.32, 0.68)
    road_poly = [
        (road_top - width * 0.16, horizon),
        (road_top + width * 0.16, horizon),
        (width * 1.12, height),
        (-width * 0.12, height),
    ]
    draw.polygon(road_poly, fill=(54, 58, 61, 255))

    left, right, center = lane_points(width, height, lateral_offset, curvature, width * 0.33)
    lane_color = (236, 232, 195, 230)
    edge_color = (245, 245, 245, 230)
    draw.line(left, fill=edge_color, width=3)
    draw.line(right, fill=edge_color, width=3)
    mid = list(zip(center, np.linspace(height * 0.42, height + 4, len(center))))
    draw_dashed_line(draw, mid, lane_color, 2)

    if obstacle:
        oy = int(height * random.uniform(0.58, 0.86))
        ox = int(width / 2 + lateral_offset * width * 0.18 + curvature * width * 0.15 + random.uniform(-12, 12))
        ow = int(width * random.uniform(0.06, 0.12))
        oh = int(height * random.uniform(0.05, 0.12))
        draw.rounded_rectangle([ox - ow, oy - oh, ox + ow, oy + oh], radius=3, fill=(55, 48, 45, 230))
        draw.rectangle([ox - ow * 0.8, oy - oh * 0.25, ox + ow * 0.8, oy + oh * 0.6], fill=(125, 38, 31, 235))

    if rain:
        for _ in range(55):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.line([(x, y), (x + random.randint(-1, 2), y + random.randint(5, 10))], fill=(210, 220, 230, 85), width=1)

    if glare:
        gx = random.randint(int(width * 0.15), int(width * 0.85))
        gy = random.randint(0, int(height * 0.25))
        draw.ellipse([gx - 18, gy - 18, gx + 18, gy + 18], fill=(255, 240, 170, 95))

    if fog:
        overlay = Image.new("RGBA", (width, height), (220, 225, 225, 55))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    arr = np.asarray(img).astype(np.int16)
    brightness = np.random.normal(0, 12)
    arr = np.clip(arr + brightness + np.random.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    if random.random() < 0.08:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.1)))
    if bad_frame:
        if random.random() < 0.5:
            img = Image.new("RGB", (width, height), (random.randint(0, 12),) * 3)
        else:
            img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), "RGB")

    steering = float(np.clip(-0.78 * curvature - 0.55 * lateral_offset + np.random.normal(0, 0.035), -1, 1))
    throttle = float(np.clip(0.72 - 0.20 * abs(curvature) - 0.32 * obstacle - 0.18 * rain - 0.16 * fog - 0.35 * traffic_density, 0, 1))
    brake = float(np.clip(0.08 + 0.62 * obstacle + 0.22 * traffic_density + 0.10 * rain - throttle * 0.08, 0, 1))

    meta = {
        "frame_id": f"frame_{idx:06d}",
        "image_path": f"images/frame_{idx:06d}.png",
        "split_hint": split_hint,
        "curvature": curvature,
        "lateral_offset": lateral_offset,
        "traffic_density": traffic_density,
        "obstacle": int(obstacle),
        "rain": int(rain),
        "fog": int(fog),
        "glare": int(glare),
        "bad_frame": int(bad_frame),
        "steering": steering,
        "throttle": throttle,
        "brake": brake,
    }
    return img, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic front-camera driving data.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    n = args.num_samples or int(cfg["num_samples"])
    width, height = int(cfg["image_width"]), int(cfg["image_height"])
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(n):
        ratio = idx / max(n - 1, 1)
        split_hint = "train" if ratio < cfg["train_ratio"] else "val" if ratio < cfg["train_ratio"] + cfg["val_ratio"] else "test"
        img, meta = generate_frame(width, height, idx, split_hint)
        img.save(args.out_dir / meta["image_path"])
        rows.append(meta)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "metadata.csv", index=False)
    print(f"Generated {len(df)} samples at {args.out_dir.resolve()}")
    print(df[["steering", "throttle", "brake", "bad_frame"]].describe().to_string())


if __name__ == "__main__":
    main()
