from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_ts_anomaly.schema import FEATURE_COLUMNS, META_COLUMNS, SOC_COLS, TARGET_COLUMNS, TEMP_COLS, VOLT_COLS


@dataclass
class WindowData:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def add_engineered_features(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["device_id", "time"]).reset_index(drop=True)

    volt = df[VOLT_COLS]
    temp = df[TEMP_COLS]
    soc = df[SOC_COLS]
    df["cell_voltage_mean"] = volt.mean(axis=1)
    df["cell_voltage_min"] = volt.min(axis=1)
    df["cell_voltage_max"] = volt.max(axis=1)
    df["cell_voltage_std"] = volt.std(axis=1)
    df["cell_voltage_delta"] = df["cell_voltage_max"] - df["cell_voltage_min"]

    df["cell_temp_mean"] = temp.mean(axis=1)
    df["cell_temp_max"] = temp.max(axis=1)
    df["cell_temp_std"] = temp.std(axis=1)
    df["cell_temp_delta"] = temp.max(axis=1) - temp.min(axis=1)

    df["cell_soc_mean"] = soc.mean(axis=1)
    df["cell_soc_min"] = soc.min(axis=1)
    df["cell_soc_std"] = soc.std(axis=1)
    df["cell_soc_delta"] = soc.max(axis=1) - soc.min(axis=1)

    deadband = float(config.get("current_deadband_a", 0.05))
    df["abs_current_a"] = df["current_a"].abs()
    df["current_direction"] = np.sign(df["current_a"]).astype(float)
    df["is_charge"] = (df["current_a"] > deadband).astype(float)
    df["is_discharge"] = (df["current_a"] < -deadband).astype(float)

    minutes = df["time"].dt.hour * 60 + df["time"].dt.minute
    phase = 2.0 * np.pi * minutes / 1440.0
    df["hour_sin"] = np.sin(phase)
    df["hour_cos"] = np.cos(phase)

    dt_hours = float(config["sample_interval_minutes"]) / 60.0
    rated_capacity = float(config["rated_capacity_ah"])
    df["throughput_ah"] = (
        df.groupby("device_id")["current_a"].transform(lambda s: s.abs().cumsum()) * dt_hours
    )
    df["throughput_norm"] = df["throughput_ah"] / rated_capacity

    grouped = df.groupby("device_id", sort=False)
    df["d_voltage_mean"] = grouped["cell_voltage_mean"].diff().fillna(0.0)
    df["d_temp_max"] = grouped["cell_temp_max"].diff().fillna(0.0)
    df["d_soc_mean"] = grouped["cell_soc_mean"].diff().fillna(0.0)

    df["is_anomaly_point"] = df["is_anomaly_point"].astype(int)
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        df[col] = df[col].astype(np.float32)
    return df


def build_windows(
    feature_df: pd.DataFrame,
    seq_len: int,
    forecast_horizon: int,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> WindowData:
    feature_columns = feature_columns or FEATURE_COLUMNS
    target_columns = target_columns or TARGET_COLUMNS
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for device_id, group in feature_df.groupby("device_id", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_columns].to_numpy(dtype=np.float32)
        max_start = len(group) - seq_len - forecast_horizon + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            target_idx = start + seq_len + forecast_horizon - 1
            x_rows.append(features[start : start + seq_len])
            y_rows.append(targets[target_idx])
            row = group.iloc[target_idx]
            meta = {col: row[col] for col in META_COLUMNS if col in group.columns}
            meta["window_start_time"] = group.iloc[start]["time"]
            meta["target_time"] = row["time"]
            meta["device_id"] = device_id
            meta_rows.append(meta)

    if not x_rows:
        raise ValueError("No windows were built. Check seq_len, forecast_horizon, and data length.")

    return WindowData(
        x=np.stack(x_rows).astype(np.float32),
        y=np.stack(y_rows).astype(np.float32),
        meta=pd.DataFrame(meta_rows),
    )


def save_feature_manifest(path: str, config: dict) -> None:
    manifest = pd.DataFrame(
        {
            "feature_columns": pd.Series(FEATURE_COLUMNS),
            "target_columns": pd.Series(TARGET_COLUMNS),
            "seq_len": int(config["seq_len"]),
            "forecast_horizon": int(config["forecast_horizon"]),
        }
    )
    manifest.to_csv(path, index=False)
