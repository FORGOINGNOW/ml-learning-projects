import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated battery discharge curves.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/data"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    discharge_df = pd.read_csv(args.data_dir / "discharge_df.csv")
    valid_df = pd.read_csv(args.data_dir / "valid_df.csv")
    discharge_plot = discharge_df.rename(
        columns={"放电时间": "time_s", "放电倍率": "c_rate", "放电电压": "voltage_v", "已放电容量": "discharged_capacity_ah"}
    )
    valid_plot = valid_df.rename(
        columns={"放电时间": "time_s", "放电倍率": "c_rate", "放电电压": "voltage_v", "已放电容量": "discharged_capacity_ah"}
    )
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 6))
    sample = discharge_plot.groupby("c_rate", group_keys=False).head(180)
    sns.lineplot(data=sample, x="time_s", y="voltage_v", hue="c_rate", palette="viridis")
    plt.title("Synthetic Discharge Curves by C-rate")
    plt.xlabel("Discharge time (s)")
    plt.ylabel("Voltage (V)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "discharge_curves_by_c_rate.png", dpi=170)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(discharge_plot["voltage_v"], bins=50, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Voltage distribution")
    sns.boxplot(data=discharge_plot, x="c_rate", y="voltage_v", ax=axes[1])
    axes[1].set_title("Voltage by C-rate")
    sns.scatterplot(data=discharge_plot.sample(min(len(discharge_plot), 2500), random_state=42), x="discharged_capacity_ah", y="voltage_v", hue="c_rate", palette="viridis", ax=axes[2], s=12)
    axes[2].set_title("Voltage vs discharged capacity")
    plt.tight_layout()
    plt.savefig(args.out_dir / "data_feature_overview.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 6))
    for name, df, alpha in [("train", discharge_plot, 0.65), ("valid_with_degradation", valid_plot, 0.95)]:
        mean_curve = df.groupby(["c_rate", "time_s"], as_index=False)["voltage_v"].mean()
        sns.lineplot(data=mean_curve, x="time_s", y="voltage_v", hue="c_rate", alpha=alpha, legend=name == "train")
    plt.title("Train Curves vs Validation Curves")
    plt.xlabel("Discharge time (s)")
    plt.ylabel("Voltage (V)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "train_valid_curve_comparison.png", dpi=170)
    plt.close()

    print(f"Saved data plots to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
