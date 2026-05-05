"""数据可视化脚本：输出标签分布、工况分布、温度和电压分布等数据检查图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_classifier.features import infer_condition
from battery_classifier.schema import SOC_COLS, TEMP_COLS, VOLT_COLS
from battery_classifier.utils import ensure_dirs, load_config, project_root

COMPARISON_FEATURES = [
    ("temp_cell_range_max", "Max temp cell spread"),
    ("temp_max_abs_step", "Max temp jump"),
    ("volt_cell_range_max", "Max voltage cell spread"),
    ("volt_max_abs_step", "Max voltage jump"),
    ("soc_cell_range_max", "Max SOC cell spread"),
    ("soc_max_abs_step", "Max SOC jump"),
    ("temp_pack_mean_slope", "Temp trend"),
    ("volt_pack_mean_slope", "Voltage trend"),
    ("soc_pack_mean_slope", "SOC trend"),
]
STATE_COLORS = {
    "normal": "#4c78a8",
    "outliers": "#e45756",
    "level_shift": "#f58518",
    "gradual_drift": "#b279a2",
    "battery_replacement": "#54a24b",
}


def _bar(series: pd.Series, title: str, output: Path, xlabel: str = "") -> None:
    counts = series.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", ax=ax, color="#4477aa")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _hist(values, title: str, output: Path, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=40, color="#66aa88", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _comparison_columns(features: pd.DataFrame) -> list[tuple[str, str]]:
    return [(col, label) for col, label in COMPARISON_FEATURES if col in features.columns]


def _state_order(features: pd.DataFrame, configured_labels: list[str] | None = None) -> list[str]:
    present = set(features["state_label"].dropna().astype(str))
    configured_labels = configured_labels or []
    ordered = []
    if "normal" in present:
        ordered.append("normal")
    for label in configured_labels:
        if label != "normal" and label in present and label not in ordered:
            ordered.append(label)
    for label in sorted(present):
        if label not in ordered:
            ordered.append(label)
    return ordered


def _state_color(label: str) -> str:
    fallback = plt.cm.tab10(abs(hash(label)) % 10)
    return STATE_COLORS.get(label, fallback)


def _add_health_group(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["health_group"] = np.where(df["state_label"].eq("normal"), "normal", "abnormal")
    return df


def _plot_unavailable(title: str, output: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _normal_abnormal_boxplots(features: pd.DataFrame, output: Path) -> None:
    cols = _comparison_columns(features)[:6]
    if not cols or features["health_group"].nunique() < 2:
        _plot_unavailable(
            "Normal vs abnormal feature distributions",
            output,
            "Need both normal and abnormal feature rows to create this plot.",
        )
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    colors = ["#4c78a8", "#e45756"]
    for ax, (col, label) in zip(axes.ravel(), cols):
        data = [
            features.loc[features["health_group"].eq("normal"), col].to_numpy(dtype=float),
            features.loc[features["health_group"].eq("abnormal"), col].to_numpy(dtype=float),
        ]
        box = ax.boxplot(data, labels=["normal", "abnormal"], patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Normal vs abnormal key feature distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _normal_abnormal_mean_gap(features: pd.DataFrame, output: Path) -> None:
    cols = _comparison_columns(features)
    if not cols or features["health_group"].nunique() < 2:
        _plot_unavailable(
            "Standardized abnormal-minus-normal feature gap",
            output,
            "Need both normal and abnormal feature rows to create this plot.",
        )
        return

    rows = []
    normal = features["health_group"].eq("normal")
    abnormal = features["health_group"].eq("abnormal")
    for col, label in cols:
        all_std = float(features[col].std(ddof=0))
        denom = all_std if all_std > 1e-12 else 1.0
        gap = (features.loc[abnormal, col].mean() - features.loc[normal, col].mean()) / denom
        rows.append((label, float(gap)))
    gap_df = pd.DataFrame(rows, columns=["feature", "gap"]).assign(abs_gap=lambda df: df["gap"].abs())
    gap_df = gap_df.sort_values("abs_gap", ascending=True).tail(8)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = np.where(gap_df["gap"] >= 0, "#e45756", "#4c78a8")
    ax.barh(gap_df["feature"], gap_df["gap"], color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Standardized abnormal-minus-normal feature gap")
    ax.set_xlabel("z-score difference")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _state_deviation_heatmap(features: pd.DataFrame, output: Path) -> None:
    cols = _comparison_columns(features)
    states = [label for label in _state_order(features) if label != "normal"]
    if not cols or not states or features["state_label"].eq("normal").sum() == 0:
        _plot_unavailable(
            "Abnormal type deviation from normal",
            output,
            "Need normal rows and at least one abnormal state to create this plot.",
        )
        return

    col_names = [col for col, _ in cols]
    labels = [label for _, label in cols]
    normal_mean = features.loc[features["state_label"].eq("normal"), col_names].mean()
    scale = features[col_names].std(ddof=0).replace(0, np.nan).fillna(1.0)
    matrix = []
    for state in states:
        state_mean = features.loc[features["state_label"].eq(state), col_names].mean()
        matrix.append(((state_mean - normal_mean) / scale).to_numpy(dtype=float))
    values = np.vstack(matrix)
    limit = max(1.0, float(np.nanmax(np.abs(values))))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    im = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title("Abnormal type deviation from normal feature mean")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels(states)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("z-score vs normal")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _all_state_feature_boxplots(features: pd.DataFrame, output: Path, label_order: list[str]) -> None:
    cols = _comparison_columns(features)[:6]
    label_order = [label for label in label_order if features["state_label"].eq(label).any()]
    if len(label_order) < 2 or not cols:
        _plot_unavailable(
            "Normal and each abnormal type feature distributions",
            output,
            "Need at least two state labels and feature rows to create this plot.",
        )
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (col, label) in zip(axes.ravel(), cols):
        data = [features.loc[features["state_label"].eq(state), col].to_numpy(dtype=float) for state in label_order]
        box = ax.boxplot(data, labels=label_order, patch_artist=True, showfliers=False)
        for patch, state in zip(box["boxes"], label_order):
            patch.set_facecolor(_state_color(state))
            patch.set_alpha(0.78)
        ax.set_title(label)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")
    fig.suptitle("Normal and each abnormal type key feature distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _choose_timeseries_devices(features: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = [col for col, _ in _comparison_columns(features)]
    normal_features = features[features["state_label"].eq("normal")]
    abnormal_features = features[~features["state_label"].eq("normal")]
    if normal_features.empty or abnormal_features.empty:
        return None, None

    normal_device = str(normal_features["devices"].iloc[0])
    if not cols:
        return normal_device, str(abnormal_features["devices"].iloc[0])

    scale = features[cols].std(ddof=0).replace(0, np.nan).fillna(1.0)
    centered = (features[cols] - features[cols].mean()) / scale
    scored = features.loc[abnormal_features.index, ["devices"]].copy()
    scored["score"] = centered.loc[abnormal_features.index].abs().mean(axis=1)
    abnormal_device = str(scored.groupby("devices")["score"].mean().sort_values(ascending=False).index[0])
    return normal_device, abnormal_device


def _choose_state_timeseries_devices(features: pd.DataFrame, label_order: list[str]) -> dict[str, str]:
    cols = [col for col, _ in _comparison_columns(features)]
    selected = {}
    if cols:
        scale = features[cols].std(ddof=0).replace(0, np.nan).fillna(1.0)
        centered = (features[cols] - features[cols].mean()) / scale
        score = centered.abs().mean(axis=1)
    else:
        score = pd.Series(0.0, index=features.index)

    for label in label_order:
        state_features = features[features["state_label"].eq(label)]
        if state_features.empty:
            continue
        if label == "normal":
            selected[label] = str(state_features["devices"].iloc[0])
            continue
        ranked = state_features[["devices"]].copy()
        ranked["score"] = score.loc[state_features.index]
        selected[label] = str(ranked.groupby("devices")["score"].mean().sort_values(ascending=False).index[0])
    return selected


def _normal_abnormal_timeseries(raw: pd.DataFrame, features: pd.DataFrame, output: Path) -> None:
    normal_device, abnormal_device = _choose_timeseries_devices(features)
    if not normal_device or not abnormal_device:
        _plot_unavailable(
            "Normal vs abnormal raw time-series example",
            output,
            "Need both normal and abnormal devices to create this plot.",
        )
        return

    subset = raw[raw["devices"].isin([normal_device, abnormal_device])].copy()
    if subset.empty:
        _plot_unavailable(
            "Normal vs abnormal raw time-series example",
            output,
            "Selected devices were not found in the raw data.",
        )
        return

    subset["time"] = pd.to_datetime(subset["time"], errors="coerce")
    subset = subset.dropna(subset=["time"]).sort_values(["devices", "time"])
    if subset.empty:
        _plot_unavailable(
            "Normal vs abnormal raw time-series example",
            output,
            "Selected rows do not contain valid timestamps.",
        )
        return

    subset["hours"] = subset.groupby("devices")["time"].transform(lambda s: (s - s.min()).dt.total_seconds() / 3600.0)
    subset["temp_pack_mean"] = subset[TEMP_COLS].mean(axis=1)
    subset["volt_pack_mean"] = subset[VOLT_COLS].mean(axis=1)
    subset["soc_pack_mean"] = subset[SOC_COLS].mean(axis=1)
    label_by_device = subset.drop_duplicates("devices").set_index("devices")["state_label"].to_dict()
    series = [
        ("temp_pack_mean", "Pack mean temperature", "deg C"),
        ("volt_pack_mean", "Pack mean voltage", "V"),
        ("soc_pack_mean", "Pack mean SOC", "%"),
    ]
    color_by_device = {normal_device: "#4c78a8", abnormal_device: "#e45756"}

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (col, title, ylabel) in zip(axes, series):
        for device in [normal_device, abnormal_device]:
            part = subset[subset["devices"].eq(device)]
            label = f"{device} ({label_by_device.get(device, 'unknown')})"
            ax.plot(part["hours"], part[col], label=label, color=color_by_device[device], linewidth=1.35, alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("hours from first sample")
    axes[0].legend(loc="best")
    fig.suptitle("Normal vs abnormal raw time-series example")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _all_state_timeseries(raw: pd.DataFrame, features: pd.DataFrame, output: Path, label_order: list[str]) -> None:
    selected = _choose_state_timeseries_devices(features, label_order)
    if len(selected) < 2:
        _plot_unavailable(
            "Normal and each abnormal type raw time-series examples",
            output,
            "Need at least two state labels to create this plot.",
        )
        return

    subset = raw[raw["devices"].isin(selected.values())].copy()
    if subset.empty:
        _plot_unavailable(
            "Normal and each abnormal type raw time-series examples",
            output,
            "Selected devices were not found in the raw data.",
        )
        return

    subset["time"] = pd.to_datetime(subset["time"], errors="coerce")
    subset = subset.dropna(subset=["time"]).sort_values(["devices", "time"])
    subset["hours"] = subset.groupby("devices")["time"].transform(lambda s: (s - s.min()).dt.total_seconds() / 3600.0)
    subset["temp_pack_mean"] = subset[TEMP_COLS].mean(axis=1)
    subset["volt_pack_mean"] = subset[VOLT_COLS].mean(axis=1)
    subset["soc_pack_mean"] = subset[SOC_COLS].mean(axis=1)
    series = [
        ("temp_pack_mean", "Pack mean temperature", "deg C"),
        ("volt_pack_mean", "Pack mean voltage", "V"),
        ("soc_pack_mean", "Pack mean SOC", "%"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    for ax, (col, title, ylabel) in zip(axes, series):
        for label in label_order:
            device = selected.get(label)
            if not device:
                continue
            part = subset[subset["devices"].eq(device)]
            ax.plot(
                part["hours"],
                part[col],
                label=f"{label} ({device})",
                color=_state_color(label),
                linewidth=1.25,
                alpha=0.88,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("hours from first sample")
    axes[0].legend(loc="best", ncol=2, fontsize=8)
    fig.suptitle("Normal and each abnormal type raw time-series examples")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create quick data QA plots.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--raw", default="data/raw/simulated_battery_readings.csv")
    parser.add_argument("--features", default="data/processed/features.csv")
    args = parser.parse_args()

    root = project_root()
    config = load_config(args.config)
    ensure_dirs(root)
    out_dir = root / "reports" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(root / args.raw)
    features = pd.read_csv(root / args.features)
    raw["condition"] = infer_condition(raw["current"], float(config["current_deadband_a"]))
    features = _add_health_group(features)
    label_order = _state_order(features, list(config.get("abnormal_states", [])))

    _bar(raw.drop_duplicates("devices")["state_label"], "Device label distribution", out_dir / "01_device_label_distribution.png")
    _bar(raw["condition"], "Raw row operating-condition distribution", out_dir / "02_condition_distribution.png")
    _bar(features["condition"], "Feature row operating-condition distribution", out_dir / "03_feature_condition_distribution.png")
    _hist(raw[TEMP_COLS].to_numpy().ravel(), "Cell temperature distribution", out_dir / "04_temperature_distribution.png", "deg C")
    _hist(raw[VOLT_COLS].to_numpy().ravel(), "Cell voltage distribution", out_dir / "05_voltage_distribution.png", "V")
    _normal_abnormal_boxplots(features, out_dir / "06_normal_abnormal_feature_boxplots.png")
    _normal_abnormal_mean_gap(features, out_dir / "07_normal_abnormal_standardized_gap.png")
    _normal_abnormal_timeseries(raw, features, out_dir / "08_normal_abnormal_timeseries_example.png")
    _state_deviation_heatmap(features, out_dir / "09_abnormal_type_deviation_heatmap.png")
    _all_state_feature_boxplots(features, out_dir / "10_all_state_feature_boxplots.png", label_order)
    _all_state_timeseries(raw, features, out_dir / "11_all_state_timeseries_examples.png", label_order)
    print(f"Saved data plots: {out_dir}")


if __name__ == "__main__":
    main()
