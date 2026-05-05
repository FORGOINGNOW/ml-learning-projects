"""特征工程模块：按工况提取电流、温度、电压、SOC 的统计和趋势特征。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery_classifier.schema import SOC_COLS, TEMP_COLS, VOLT_COLS

META_COLUMNS = ["split", "devices", "state_label", "date", "condition"]


def infer_condition(current: pd.Series, deadband: float) -> pd.Series:
    values = current.to_numpy(dtype=float)
    conditions = np.full(len(values), "idle", dtype=object)
    conditions[values > deadband] = "charge"
    conditions[values < -deadband] = "discharge"
    return pd.Series(conditions, index=current.index)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [col for col in features.columns if col not in META_COLUMNS]


def validate_raw_schema(raw: pd.DataFrame) -> None:
    required = ["time", "devices", "current"] + TEMP_COLS + VOLT_COLS + SOC_COLS
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1))


def _safe_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    if np.nanstd(values) < 1e-12:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values))
    return float(np.polyfit(x, values, 1)[0])


def _add_series_stats(row: dict, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    row[f"{name}_mean"] = float(np.nanmean(values))
    row[f"{name}_std"] = _safe_std(values)
    row[f"{name}_min"] = float(np.nanmin(values))
    row[f"{name}_max"] = float(np.nanmax(values))
    row[f"{name}_range"] = float(np.nanmax(values) - np.nanmin(values))
    row[f"{name}_last_minus_first"] = float(values[-1] - values[0]) if len(values) else 0.0
    row[f"{name}_slope"] = _safe_slope(values)


def _family_name(columns: list[str]) -> str:
    if columns == TEMP_COLS:
        return "temp"
    if columns == VOLT_COLS:
        return "volt"
    if columns == SOC_COLS:
        return "soc"
    return "sensor"


def _add_cell_family_features(row: dict, group: pd.DataFrame, columns: list[str]) -> None:
    family = _family_name(columns)
    values = group[columns].to_numpy(dtype=float)
    pack_mean = values.mean(axis=1)
    cell_range = values.max(axis=1) - values.min(axis=1)
    cell_std = values.std(axis=1)

    _add_series_stats(row, f"{family}_pack_mean", pack_mean)
    _add_series_stats(row, f"{family}_cell_range", cell_range)
    _add_series_stats(row, f"{family}_cell_std", cell_std)

    if len(values) > 1:
        step = np.abs(np.diff(values, axis=0))
        row[f"{family}_max_abs_step"] = float(np.max(step))
        row[f"{family}_mean_abs_step"] = float(np.mean(step))
    else:
        row[f"{family}_max_abs_step"] = 0.0
        row[f"{family}_mean_abs_step"] = 0.0

    for col in columns:
        _add_series_stats(row, col, group[col].to_numpy(dtype=float))


def _features_for_group(keys: tuple, group: pd.DataFrame) -> dict:
    split, devices, state_label, date, condition = keys
    row: dict[str, float | str] = {
        "split": split,
        "devices": devices,
        "state_label": state_label,
        "date": date,
        "condition": condition,
        "sample_count": float(len(group)),
    }
    elapsed_minutes = (
        (group["time"].iloc[-1] - group["time"].iloc[0]).total_seconds() / 60.0
        if len(group) > 1
        else 0.0
    )
    row["duration_hours"] = float(elapsed_minutes / 60.0)
    current = group["current"].to_numpy(dtype=float)
    _add_series_stats(row, "current", current)
    row["abs_current_mean"] = float(np.mean(np.abs(current)))
    row["discharge_rate_a"] = float(np.mean(np.abs(current[current < 0]))) if np.any(current < 0) else 0.0
    row["charge_rate_a"] = float(np.mean(current[current > 0])) if np.any(current > 0) else 0.0

    for columns in [TEMP_COLS, VOLT_COLS, SOC_COLS]:
        _add_cell_family_features(row, group, columns)

    temp_values = group[TEMP_COLS].to_numpy(dtype=float)
    volt_values = group[VOLT_COLS].to_numpy(dtype=float)
    row["temp_out_of_range_rate"] = float(np.mean((temp_values < 20.0) | (temp_values > 60.0)))
    row["volt_over_limit_rate"] = float(np.mean(volt_values > 3.5))
    return row


def build_feature_table(raw: pd.DataFrame, current_deadband: float) -> pd.DataFrame:
    validate_raw_schema(raw)
    df = raw.copy()
    if "split" not in df.columns:
        df["split"] = "inference"
    if "state_label" not in df.columns:
        df["state_label"] = "unknown"

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        bad_count = int(df["time"].isna().sum())
        raise ValueError(f"{bad_count} rows have invalid time values.")
    df["date"] = df["time"].dt.date.astype(str)
    df["condition"] = infer_condition(df["current"], current_deadband)
    df = df[df["condition"] != "idle"].copy()
    if df.empty:
        raise ValueError("No charge/discharge rows found after current deadband filtering.")

    group_cols = ["split", "devices", "state_label", "date", "condition"]
    rows = [_features_for_group(keys, group) for keys, group in df.groupby(group_cols, sort=False)]
    features = pd.DataFrame(rows)
    numeric_cols = feature_columns(features)
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def save_feature_table(raw_path: Path, output_path: Path, current_deadband: float) -> Path:
    raw = pd.read_csv(raw_path)
    features = build_feature_table(raw, current_deadband)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path
