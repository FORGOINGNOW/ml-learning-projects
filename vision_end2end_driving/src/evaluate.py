import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from dataset import DrivingDataset
from model import EndToEndDrivingNet


TARGETS = ["steering", "throttle", "brake"]


def predict(model, loader, device):
    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
            ys.append(y.numpy())
    return np.vstack(ys), np.vstack(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end driving model.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/e2e_cnn/best_model.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/model"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = DrivingDataset(args.processed_dir / "test.csv", args.raw_dir, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = EndToEndDrivingNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    y_true, y_pred = predict(model, loader, device)
    metrics = []
    for i, name in enumerate(TARGETS):
        metrics.append(
            {
                "target": name,
                "mae": mean_absolute_error(y_true[:, i], y_pred[:, i]),
                "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
                "r2": r2_score(y_true[:, i], y_pred[:, i]),
            }
        )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.out_dir / "test_metrics.csv", index=False)

    pred_df = ds.df.copy()
    for i, name in enumerate(TARGETS):
        pred_df[f"pred_{name}"] = y_pred[:, i]
        pred_df[f"abs_error_{name}"] = np.abs(y_pred[:, i] - y_true[:, i])
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (ax, name) in enumerate(zip(axes, TARGETS)):
        sns.scatterplot(x=y_true[:, i], y=y_pred[:, i], s=16, alpha=0.55, ax=ax)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name}: predicted vs true")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    plt.tight_layout()
    plt.savefig(args.out_dir / "prediction_scatter.png", dpi=160)
    plt.close()

    err_long = pred_df[[f"abs_error_{t}" for t in TARGETS]].melt(var_name="target", value_name="absolute_error")
    err_long["target"] = err_long["target"].str.replace("abs_error_", "", regex=False)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=err_long, x="target", y="absolute_error")
    plt.title("Absolute Error Distribution")
    plt.tight_layout()
    plt.savefig(args.out_dir / "absolute_error_boxplot.png", dpi=160)
    plt.close()

    worst = pred_df.sort_values("abs_error_steering", ascending=False).head(16).reset_index(drop=True)
    thumb_w, thumb_h = 160, 96
    canvas = Image.new("RGB", (thumb_w * 4, (thumb_h + 34) * 4), "white")
    for i, row in worst.iterrows():
        img = Image.open(args.raw_dir / row["image_path"]).convert("RGB").resize((thumb_w, thumb_h))
        draw = ImageDraw.Draw(img)
        text1 = f"true s={row.steering:+.2f} p={row.pred_steering:+.2f}"
        text2 = f"t={row.pred_throttle:.2f} b={row.pred_brake:.2f}"
        draw.rectangle([0, thumb_h - 26, thumb_w, thumb_h], fill=(0, 0, 0))
        draw.text((4, thumb_h - 25), text1, fill=(255, 255, 255))
        draw.text((4, thumb_h - 13), text2, fill=(255, 255, 255))
        canvas.paste(img, ((i % 4) * thumb_w, (i // 4) * (thumb_h + 34)))
    canvas.save(args.out_dir / "worst_steering_examples.png")

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "checkpoint_val_loss": float(checkpoint.get("val_loss", np.nan)),
                "metrics": metrics_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Saved evaluation reports to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
