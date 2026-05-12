from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from battery_ts_anomaly.schema import TARGET_COLUMNS


def save_training_curve(history_path: Path, out_dir: Path) -> None:
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="validation")
    ax.set_title("LSTM training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Smooth L1 loss (scaled targets)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=160)
    plt.close()


def save_score_distribution(pred: pd.DataFrame, out_dir: Path) -> None:
    eval_df = pred[pred["split"].isin(["valid", "test"])]
    normal = eval_df[eval_df["is_anomaly_point"] == 0]["anomaly_score"]
    abnormal = eval_df[eval_df["is_anomaly_point"] == 1]["anomaly_score"]
    threshold = float(pred["threshold"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(normal, bins=60, alpha=0.68, density=True, label="normal windows")
    ax.hist(abnormal, bins=60, alpha=0.58, density=True, label="abnormal windows")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.4, label=f"threshold={threshold:.3f}")
    ax.set_title("Residual anomaly score distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution.png", dpi=160)
    plt.close()


def save_confusion_matrix(pred: pd.DataFrame, out_dir: Path) -> None:
    test = pred[pred["split"] == "test"]
    y_true = test["is_anomaly_point"].astype(int)
    y_pred = test["is_detected"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "detected"])
    ax.set_yticks([0, 1], labels=["normal", "anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test confusion matrix")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "test_confusion_matrix.png", dpi=160)
    plt.close()


def save_residual_contributors(pred: pd.DataFrame, out_dir: Path) -> None:
    eval_df = pred[(pred["split"].isin(["valid", "test"])) & (pred["is_anomaly_point"] == 1)]
    values = []
    for col in TARGET_COLUMNS:
        values.append(eval_df[f"abs_norm_residual_{col}"].mean())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(TARGET_COLUMNS, values, color="tab:purple", alpha=0.76)
    ax.set_title("Mean normalized residual by predicted metric")
    ax.set_ylabel("Mean abs normalized residual")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_contributors.png", dpi=160)
    plt.close()


def save_timeline_examples(pred: pd.DataFrame, out_dir: Path) -> None:
    eval_df = pred[(pred["split"] == "test") & (pred["state_label"] != "normal")].copy()
    device_ids = list(eval_df.groupby("device_id")["anomaly_score"].max().sort_values(ascending=False).head(3).index)
    if not device_ids:
        return
    fig, axes = plt.subplots(len(device_ids), 1, figsize=(11, 3.2 * len(device_ids)), sharex=False)
    if len(device_ids) == 1:
        axes = [axes]
    threshold = float(pred["threshold"].iloc[0])
    for ax, device_id in zip(axes, device_ids):
        group = eval_df[eval_df["device_id"] == device_id].copy()
        group["target_time"] = pd.to_datetime(group["target_time"])
        ax.plot(group["target_time"], group["anomaly_score"], color="tab:blue", linewidth=1.15, label="score")
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.1, label="threshold")
        abnormal = group[group["is_anomaly_point"] == 1]
        ax.axvspan(abnormal["target_time"].min(), abnormal["target_time"].max(), color="orange", alpha=0.16)
        detected = group[group["is_detected"]]
        ax.scatter(detected["target_time"], detected["anomaly_score"], s=12, color="tab:red", label="detected")
        ax.set_title(f"{device_id} / {group['state_label'].iloc[0]}")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Target time")
    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_timeline_examples.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model visualizations.")
    parser.add_argument("--predictions", type=Path, default=Path("reports/model/window_predictions.csv"))
    parser.add_argument("--history", type=Path, default=Path("runs/train_history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions, low_memory=False)
    save_training_curve(args.history, args.output_dir)
    save_score_distribution(pred, args.output_dir)
    save_confusion_matrix(pred, args.output_dir)
    save_residual_contributors(pred, args.output_dir)
    save_timeline_examples(pred, args.output_dir)
    print(f"Saved model plots under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
