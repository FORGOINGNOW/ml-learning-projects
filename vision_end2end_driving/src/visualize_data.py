import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw


TARGETS = ["steering", "throttle", "brake"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize driving dataset features and labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_all.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/data"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, TARGETS):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(f"{col} distribution")
    plt.tight_layout()
    plt.savefig(args.out_dir / "label_distributions.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(len(df), 1500), random_state=42), x="curvature", y="steering", hue="obstacle", alpha=0.65)
    plt.title("Steering vs. Road Curvature")
    plt.tight_layout()
    plt.savefig(args.out_dir / "steering_vs_curvature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    condition_rate = df[["rain", "fog", "glare", "obstacle"]].mean().sort_values(ascending=False)
    sns.barplot(x=condition_rate.index, y=condition_rate.values, color="#72B7B2")
    plt.ylabel("rate")
    plt.title("Scenario Condition Rates")
    plt.tight_layout()
    plt.savefig(args.out_dir / "scenario_rates.png", dpi=160)
    plt.close()

    sample = df.sample(min(24, len(df)), random_state=7).reset_index(drop=True)
    thumb_w, thumb_h = 160, 96
    canvas = Image.new("RGB", (thumb_w * 4, (thumb_h + 28) * 6), "white")
    for i, row in sample.iterrows():
        img = Image.open(args.raw_dir / row["image_path"]).convert("RGB").resize((thumb_w, thumb_h))
        draw = ImageDraw.Draw(img)
        text = f"s={row.steering:+.2f} t={row.throttle:.2f} b={row.brake:.2f}"
        draw.rectangle([0, thumb_h - 15, thumb_w, thumb_h], fill=(0, 0, 0))
        draw.text((4, thumb_h - 14), text, fill=(255, 255, 255))
        x = (i % 4) * thumb_w
        y = (i // 4) * (thumb_h + 28)
        canvas.paste(img, (x, y))
    canvas.save(args.out_dir / "sample_grid.png")
    print(f"Saved visualizations to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
