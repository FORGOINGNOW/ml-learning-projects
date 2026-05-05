"""仿真模块：生成满足电池业务约束和异常状态模式的合成 BMS 数据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from battery_classifier.schema import SOC_COLS, TEMP_COLS, VOLT_COLS


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    split: str
    state_label: str
    rated_capacity_ah: float
    cell_capacity_scale: np.ndarray
    temp_offset: np.ndarray
    volt_offset: np.ndarray
    initial_soc: np.ndarray


def balanced_labels(total: int, labels: list[str], rng: np.random.Generator) -> list[str]:
    repeated = labels * (total // len(labels))
    repeated.extend(labels[: total - len(repeated)])
    repeated = list(repeated)
    rng.shuffle(repeated)
    return repeated


def build_device_specs(config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    labels = list(config["abnormal_states"])
    split_sizes = {
        "train": int(config["train_devices"]),
        "valid": int(config["valid_devices"]),
        "test": int(config["test_devices"]),
    }
    rows = []
    device_num = 1
    for split, size in split_sizes.items():
        split_labels = balanced_labels(size, labels, rng)
        for state_label in split_labels:
            rows.append(
                {
                    "devices": f"DEV_{device_num:04d}",
                    "split": split,
                    "state_label": state_label,
                }
            )
            device_num += 1
    return pd.DataFrame(rows)


def _current_profile(points: int, sample_interval_minutes: int, rng: np.random.Generator) -> np.ndarray:
    minutes = np.arange(points) * sample_interval_minutes
    hour = (minutes % 1440) / 60.0
    discharge = (hour >= 7.0) & (hour < 20.0)
    base = np.where(discharge, -0.50, 0.68)
    daily_wave = 0.04 * np.sin(2 * np.pi * minutes / 1440.0)
    noise = rng.normal(0.0, 0.035, size=points)
    current = base + daily_wave + noise
    current[discharge] = np.clip(current[discharge], -0.75, -0.25)
    current[~discharge] = np.clip(current[~discharge], 0.25, 0.95)
    return current


def _soc_from_current(
    current: np.ndarray,
    initial_soc: np.ndarray,
    capacity_ah: np.ndarray,
    sample_interval_minutes: int,
) -> np.ndarray:
    dt_hours = sample_interval_minutes / 60.0
    soc = np.zeros((len(current), len(initial_soc)), dtype=float)
    soc[0] = initial_soc
    for idx in range(1, len(current)):
        efficiency = 0.97 if current[idx] > 0 else 1.0
        delta_pct = current[idx] * dt_hours * efficiency / capacity_ah * 100.0
        soc[idx] = np.clip(soc[idx - 1] + delta_pct, 8.0, 100.0)
    return soc


def _base_voltage(soc: np.ndarray, current: np.ndarray, volt_offset: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ocv = 2.92 + 0.00515 * soc - 0.000004 * np.square(100.0 - soc)
    polarization = current[:, None] * 0.035
    noise = rng.normal(0.0, 0.006, size=soc.shape)
    volt = ocv + polarization + volt_offset[None, :] + noise
    return np.clip(volt, 2.70, 3.50)


def _base_temperature(
    current: np.ndarray,
    sample_interval_minutes: int,
    temp_offset: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    minutes = np.arange(len(current)) * sample_interval_minutes
    hour = (minutes % 1440) / 60.0
    ambient = 27.0 + 3.0 * np.sin(2 * np.pi * (hour - 13.0) / 24.0)
    heating = 4.2 * np.abs(current)
    noise = rng.normal(0.0, 0.35, size=(len(current), len(temp_offset)))
    temp = ambient[:, None] + heating[:, None] + temp_offset[None, :] + noise
    return np.clip(temp, 20.0, 60.0)


def _apply_outliers(
    temp: np.ndarray,
    volt: np.ndarray,
    soc: np.ndarray,
    rng: np.random.Generator,
) -> None:
    points, cells = temp.shape
    event_count = max(8, points // 50)
    row_idx = rng.choice(points, size=event_count, replace=False)
    cell_idx = rng.integers(0, cells, size=event_count)
    temp[row_idx, cell_idx] += rng.uniform(8.0, 22.0, size=event_count)
    volt[row_idx, cell_idx] += rng.choice([-1.0, 1.0], size=event_count) * rng.uniform(0.04, 0.13, size=event_count)
    soc[row_idx, cell_idx] += rng.choice([-1.0, 1.0], size=event_count) * rng.uniform(4.0, 12.0, size=event_count)


def _apply_level_shift(
    temp: np.ndarray,
    volt: np.ndarray,
    soc: np.ndarray,
    rng: np.random.Generator,
) -> None:
    points, cells = temp.shape
    start = int(rng.uniform(0.25, 0.65) * points)
    cell = int(rng.integers(0, cells))
    temp[start:, cell] += rng.uniform(3.5, 7.5)
    volt[start:, cell] -= rng.uniform(0.045, 0.11)
    soc[start:, cell] -= rng.uniform(3.5, 8.0)


def _apply_gradual_drift(
    temp: np.ndarray,
    volt: np.ndarray,
    soc: np.ndarray,
    rng: np.random.Generator,
) -> None:
    points, cells = temp.shape
    cell = int(rng.integers(0, cells))
    ramp = np.linspace(0.0, 1.0, points)
    temp[:, cell] += ramp * rng.uniform(4.0, 9.0)
    volt[:, cell] -= ramp * rng.uniform(0.06, 0.14)
    soc[:, cell] -= ramp * rng.uniform(5.0, 12.0)


def _apply_battery_replacement(
    temp: np.ndarray,
    volt: np.ndarray,
    soc: np.ndarray,
    rng: np.random.Generator,
) -> None:
    points, cells = temp.shape
    start = int(rng.uniform(0.40, 0.70) * points)
    cell = int(rng.integers(0, cells))
    other_cells = [idx for idx in range(cells) if idx != cell]
    pack_soc = np.mean(soc[start:, other_cells], axis=1)
    reset_delta = rng.uniform(7.0, 16.0) * rng.choice([-1.0, 1.0])
    soc[start:, cell] = pack_soc + reset_delta + rng.normal(0.0, 0.8, size=points - start)
    volt[start:, cell] += np.sign(reset_delta) * rng.uniform(0.035, 0.09)
    temp[start:, cell] += rng.uniform(-2.5, 1.0)
    edge = min(points, start + 4)
    volt[start:edge, cell] += rng.choice([-1.0, 1.0]) * rng.uniform(0.07, 0.12)


def _apply_state(
    state_label: str,
    temp: np.ndarray,
    volt: np.ndarray,
    soc: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if state_label == "outliers":
        _apply_outliers(temp, volt, soc, rng)
    elif state_label == "level_shift":
        _apply_level_shift(temp, volt, soc, rng)
    elif state_label == "gradual_drift":
        _apply_gradual_drift(temp, volt, soc, rng)
    elif state_label == "battery_replacement":
        _apply_battery_replacement(temp, volt, soc, rng)

    np.clip(temp, 20.0, 60.0, out=temp)
    np.clip(volt, 2.70, 3.50, out=volt)
    np.clip(soc, 0.0, 100.0, out=soc)


def _make_device_spec(row: pd.Series, rated_capacity_ah: float, rng: np.random.Generator) -> DeviceSpec:
    cells = len(TEMP_COLS)
    return DeviceSpec(
        device_id=str(row["devices"]),
        split=str(row["split"]),
        state_label=str(row["state_label"]),
        rated_capacity_ah=rated_capacity_ah,
        cell_capacity_scale=rng.normal(1.0, 0.018, size=cells),
        temp_offset=rng.normal(0.0, 0.7, size=cells),
        volt_offset=rng.normal(0.0, 0.012, size=cells),
        initial_soc=np.clip(rng.normal(62.0, 9.0, size=cells), 30.0, 90.0),
    )


def simulate_readings(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    metadata = build_device_specs(config)
    points_per_day = int(24 * 60 / int(config["sample_interval_minutes"]))
    start_date = pd.Timestamp("2026-01-01")
    frames: list[pd.DataFrame] = []

    for _, row in metadata.iterrows():
        spec = _make_device_spec(row, float(config["rated_capacity_ah"]), rng)
        capacity = spec.rated_capacity_ah * spec.cell_capacity_scale
        initial_soc = spec.initial_soc.copy()
        for day in range(int(config["days_per_device"])):
            day_start = start_date + pd.Timedelta(days=day)
            times = pd.date_range(day_start, periods=points_per_day, freq=f"{config['sample_interval_minutes']}min")
            current = _current_profile(points_per_day, int(config["sample_interval_minutes"]), rng)
            soc = _soc_from_current(current, initial_soc, capacity, int(config["sample_interval_minutes"]))
            volt = _base_voltage(soc, current, spec.volt_offset, rng)
            temp = _base_temperature(current, int(config["sample_interval_minutes"]), spec.temp_offset, rng)
            _apply_state(spec.state_label, temp, volt, soc, rng)

            data = {
                "time": times.astype(str),
                "devices": spec.device_id,
                "current": current,
                "split": spec.split,
                "state_label": spec.state_label,
            }
            for idx, col in enumerate(TEMP_COLS):
                data[col] = temp[:, idx]
            for idx, col in enumerate(VOLT_COLS):
                data[col] = volt[:, idx]
            for idx, col in enumerate(SOC_COLS):
                data[col] = soc[:, idx]
            frames.append(pd.DataFrame(data))
            initial_soc = soc[-1]

    readings = pd.concat(frames, ignore_index=True)
    ordered_cols = ["time", "devices"] + TEMP_COLS + VOLT_COLS + SOC_COLS + ["current", "split", "state_label"]
    return readings[ordered_cols], metadata


def save_simulation(config: dict, root: Path) -> tuple[Path, Path]:
    readings, metadata = simulate_readings(config)
    raw_path = root / "data" / "raw" / "simulated_battery_readings.csv"
    metadata_path = root / "data" / "raw" / "device_metadata.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    readings.to_csv(raw_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return raw_path, metadata_path
