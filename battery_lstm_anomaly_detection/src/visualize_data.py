from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_state_distribution(metadata: pd.DataFrame, out_dir: Path) -> None:
    counts = metadata.groupby(["split", "state_label"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(10, 5), width=0.82)
    ax.set_title("Device state distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Devices")
    ax.legend(title="State", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "device_state_distribution.png", dpi=160)
    plt.close()


def save_feature_distribution(features: pd.DataFrame, out_dir: Path) -> None:
    cols = ["cell_voltage_delta", "cell_temp_max", "cell_soc_delta", "abs_current_a"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    normal = features[features["is_anomaly_point"] == 0]
    abnormal = features[features["is_anomaly_point"] == 1]
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(normal[col], bins=45, alpha=0.65, label="normal", density=True)
        if len(abnormal) > 0:
            ax.hist(abnormal[col], bins=45, alpha=0.55, label="abnormal", density=True)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Engineered feature distributions")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_distributions.png", dpi=160)
    plt.close()


def _plot_device(axs: list[plt.Axes], group: pd.DataFrame, title: str) -> None:
    x = pd.to_datetime(group["time"])
    metrics = ["cell_voltage_min", "cell_temp_max", "cell_soc_delta"]
    colors = ["tab:blue", "tab:red", "tab:green"]
    for ax, metric, color in zip(axs, metrics, colors):
        ax.plot(x, group[metric], color=color, linewidth=1.1)
        if group["is_anomaly_point"].max() > 0:
            abnormal = group[group["is_anomaly_point"] == 1]
            ax.axvspan(pd.to_datetime(abnormal["time"]).min(), pd.to_datetime(abnormal["time"]).max(), color="orange", alpha=0.16)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.2)
    axs[0].set_title(title)
    axs[-1].set_xlabel("Time")


def save_timeseries_examples(features: pd.DataFrame, out_dir: Path) -> None:
    eval_df = features[features["split"].isin(["valid", "test"])].copy()
    normal_id = eval_df.loc[eval_df["state_label"] == "normal", "device_id"].iloc[0]
    abnormal_id = eval_df.loc[eval_df["state_label"] != "normal", "device_id"].iloc[0]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=False)
    normal = eval_df[eval_df["device_id"] == normal_id]
    abnormal = eval_df[eval_df["device_id"] == abnormal_id]
    _plot_device(list(axes[:, 0]), normal, f"Normal example: {normal_id}")
    _plot_device(list(axes[:, 1]), abnormal, f"Abnormal example: {abnormal_id} / {abnormal['state_label'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(out_dir / "timeseries_examples.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create data visualizations.")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/device_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features, parse_dates=["time"], low_memory=False)
    metadata = pd.read_csv(args.metadata)
    save_state_distribution(metadata, args.output_dir)
    save_feature_distribution(features, args.output_dir)
    save_timeseries_examples(features, args.output_dir)
    print(f"Saved data plots under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
