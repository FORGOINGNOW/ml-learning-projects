from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from battery_ts_anomaly.schema import SOC_COLS, TEMP_COLS, VOLT_COLS


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    split: str
    state_label: str
    capacity_ah: np.ndarray
    temp_offset: np.ndarray
    volt_offset: np.ndarray
    initial_soc: np.ndarray


def balanced_labels(total: int, labels: list[str], rng: np.random.Generator) -> list[str]:
    repeated = list(labels) * (total // len(labels))
    repeated.extend(labels[: total - len(repeated)])
    rng.shuffle(repeated)
    return repeated


def build_device_metadata(config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed"]))
    rows: list[dict] = []
    device_num = 1
    split_sizes = {
        "train": int(config["train_devices"]),
        "valid": int(config["valid_devices"]),
        "test": int(config["test_devices"]),
    }
    eval_states = list(config["evaluation_states"])

    for split, size in split_sizes.items():
        labels = ["normal"] * size if split == "train" else balanced_labels(size, eval_states, rng)
        for label in labels:
            rows.append(
                {
                    "device_id": f"BAT_{device_num:04d}",
                    "split": split,
                    "state_label": label,
                }
            )
            device_num += 1
    return pd.DataFrame(rows)


def _make_device_spec(row: pd.Series, rated_capacity_ah: float, n_cells: int, rng: np.random.Generator) -> DeviceSpec:
    capacity_scale = rng.normal(1.0, 0.018, size=n_cells)
    return DeviceSpec(
        device_id=str(row["device_id"]),
        split=str(row["split"]),
        state_label=str(row["state_label"]),
        capacity_ah=rated_capacity_ah * capacity_scale,
        temp_offset=rng.normal(0.0, 0.65, size=n_cells),
        volt_offset=rng.normal(0.0, 0.010, size=n_cells),
        initial_soc=np.clip(rng.normal(68.0, 8.0, size=n_cells), 38.0, 88.0),
    )


def _current_profile(points: int, sample_interval_minutes: int, rng: np.random.Generator) -> np.ndarray:
    minutes = np.arange(points) * sample_interval_minutes
    hour = (minutes % 1440) / 60.0
    daytime_discharge = (hour >= 7.0) & (hour < 20.0)
    evening_charge = (hour >= 20.0) & (hour < 24.0)
    night_charge = hour < 5.5
    base = np.where(daytime_discharge, -0.82, np.where(evening_charge | night_charge, 0.62, -0.05))
    daily_wave = 0.10 * np.sin(2 * np.pi * minutes / 1440.0)
    load_wave = 0.08 * np.sin(2 * np.pi * minutes / 180.0)
    noise = rng.normal(0.0, 0.045, size=points)
    current = base + daily_wave + load_wave + noise
    return np.clip(current, -1.35, 1.05)


def _soc_from_current(
    current: np.ndarray,
    initial_soc: np.ndarray,
    capacity_ah: np.ndarray,
    sample_interval_minutes: int,
) -> np.ndarray:
    dt_hours = sample_interval_minutes / 60.0
    soc = np.zeros((len(current), len(initial_soc)), dtype=np.float32)
    soc[0] = initial_soc
    for idx in range(1, len(current)):
        efficiency = 0.965 if current[idx] > 0 else 1.0
        delta_pct = current[idx] * dt_hours * efficiency / capacity_ah * 100.0
        soc[idx] = np.clip(soc[idx - 1] + delta_pct, 5.0, 98.0)
    return soc


def _base_voltage(
    soc: np.ndarray,
    current: np.ndarray,
    capacity_ah: np.ndarray,
    volt_offset: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    soc_frac = np.clip(soc / 100.0, 0.0, 1.0)
    ocv = 3.02 + 0.86 * soc_frac + 0.08 * np.tanh((soc - 55.0) / 18.0)
    resistance = 0.032 + 0.010 * (capacity_ah.mean() / capacity_ah - 1.0)
    polarization = current[:, None] * resistance[None, :]
    noise = rng.normal(0.0, 0.0045, size=soc.shape)
    voltage = ocv + polarization + volt_offset[None, :] + noise
    return np.clip(voltage, 2.75, 4.20).astype(np.float32)


def _base_temperature(
    current: np.ndarray,
    sample_interval_minutes: int,
    temp_offset: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    minutes = np.arange(len(current)) * sample_interval_minutes
    hour = (minutes % 1440) / 60.0
    ambient = 27.5 + 4.2 * np.sin(2 * np.pi * (hour - 14.0) / 24.0)
    heating = 2.4 * np.abs(current) + 0.6 * np.square(current)
    noise = rng.normal(0.0, 0.28, size=(len(current), len(temp_offset)))
    temp = ambient[:, None] + heating[:, None] + temp_offset[None, :] + noise
    return np.clip(temp, 18.0, 65.0).astype(np.float32), ambient.astype(np.float32)


def _ramp(length: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def _apply_voltage_sag(
    volt: np.ndarray,
    temp: np.ndarray,
    soc: np.ndarray,
    current: np.ndarray,
    start: int,
    cell: int,
    rng: np.random.Generator,
) -> None:
    ramp = _ramp(len(current) - start)
    discharge_load = np.clip(-current[start:], 0.0, None)
    sag = rng.uniform(0.075, 0.17) * (0.60 + discharge_load) * (0.45 + 0.55 * ramp)
    volt[start:, cell] -= sag
    temp[start:, cell] += rng.uniform(0.8, 2.2) * ramp
    soc[start:, cell] -= rng.uniform(1.0, 3.5) * ramp


def _apply_thermal_rise(
    volt: np.ndarray,
    temp: np.ndarray,
    current: np.ndarray,
    start: int,
    cell: int,
    rng: np.random.Generator,
) -> None:
    ramp = _ramp(len(current) - start)
    heat = rng.uniform(7.0, 16.0) * ramp + rng.uniform(1.5, 4.0) * np.abs(current[start:])
    temp[start:, cell] += heat
    volt[start:, cell] -= rng.uniform(0.005, 0.035) * ramp


def _apply_soc_sensor_drift(
    soc: np.ndarray,
    start: int,
    cell: int,
    rng: np.random.Generator,
) -> None:
    ramp = _ramp(len(soc) - start)
    drift = rng.choice([-1.0, 1.0]) * rng.uniform(14.0, 26.0)
    soc[start:, cell] += drift * (0.35 + 0.65 * ramp)


def _apply_cell_imbalance(
    volt: np.ndarray,
    temp: np.ndarray,
    soc: np.ndarray,
    start: int,
    cell: int,
    rng: np.random.Generator,
) -> None:
    ramp = _ramp(len(soc) - start)
    severity = 0.30 + 0.70 * ramp
    soc[start:, cell] -= rng.uniform(10.0, 24.0) * severity
    volt[start:, cell] -= rng.uniform(0.065, 0.16) * severity
    temp[start:, cell] += rng.uniform(2.0, 5.5) * ramp


def _apply_resistance_growth(
    volt: np.ndarray,
    temp: np.ndarray,
    current: np.ndarray,
    start: int,
    cell: int,
    rng: np.random.Generator,
) -> None:
    ramp = _ramp(len(current) - start)
    extra_r = rng.uniform(0.07, 0.16)
    volt[start:, cell] += current[start:] * extra_r * ramp
    temp[start:, cell] += np.abs(current[start:]) * rng.uniform(4.0, 9.0) * ramp


def _apply_state(
    state_label: str,
    volt: np.ndarray,
    temp: np.ndarray,
    soc: np.ndarray,
    current: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, str]:
    if state_label == "normal":
        return -1, ""

    points, cells = volt.shape
    start = int(rng.uniform(0.32, 0.62) * points)
    cell = int(rng.integers(0, cells))

    if state_label == "voltage_sag":
        _apply_voltage_sag(volt, temp, soc, current, start, cell, rng)
    elif state_label == "thermal_rise":
        _apply_thermal_rise(volt, temp, current, start, cell, rng)
    elif state_label == "soc_sensor_drift":
        _apply_soc_sensor_drift(soc, start, cell, rng)
    elif state_label == "cell_imbalance":
        _apply_cell_imbalance(volt, temp, soc, start, cell, rng)
    elif state_label == "resistance_growth":
        _apply_resistance_growth(volt, temp, current, start, cell, rng)
    else:
        raise ValueError(f"Unknown state_label: {state_label}")

    np.clip(volt, 2.65, 4.25, out=volt)
    np.clip(temp, 18.0, 70.0, out=temp)
    np.clip(soc, 0.0, 100.0, out=soc)
    return start, f"cell_{cell + 1}"


def _actionable_label_start(state_label: str, onset_idx: int, points: int) -> int:
    if onset_idx < 0:
        return -1
    remaining = points - onset_idx
    fractions = {
        "thermal_rise": 0.03,
        "resistance_growth": 0.07,
        "voltage_sag": 0.08,
        "cell_imbalance": 0.08,
        "soc_sensor_drift": 0.05,
    }
    offset = int(remaining * fractions.get(state_label, 0.12))
    return min(points - 1, onset_idx + max(2, offset))


def simulate_readings(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    n_cells = int(config["n_cells"])
    if n_cells != 8:
        raise ValueError("This demo keeps the existing project convention of 8 BMS cells.")

    metadata = build_device_metadata(config)
    points_per_day = int(24 * 60 / int(config["sample_interval_minutes"]))
    total_points = points_per_day * int(config["days_per_device"])
    start_date = pd.Timestamp("2026-01-01")
    frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for _, row in metadata.iterrows():
        spec = _make_device_spec(row, float(config["rated_capacity_ah"]), n_cells, rng)
        times = pd.date_range(start_date, periods=total_points, freq=f"{config['sample_interval_minutes']}min")
        current = _current_profile(total_points, int(config["sample_interval_minutes"]), rng).astype(np.float32)
        soc = _soc_from_current(current, spec.initial_soc, spec.capacity_ah, int(config["sample_interval_minutes"]))
        volt = _base_voltage(soc, current, spec.capacity_ah, spec.volt_offset, rng)
        temp, ambient = _base_temperature(current, int(config["sample_interval_minutes"]), spec.temp_offset, rng)
        fault_onset_idx, fault_cell = _apply_state(spec.state_label, volt, temp, soc, current, rng)
        anomaly_start_idx = _actionable_label_start(spec.state_label, fault_onset_idx, total_points)
        is_anomaly = np.zeros(total_points, dtype=bool)
        if anomaly_start_idx >= 0:
            is_anomaly[anomaly_start_idx:] = True
        anomaly_start_time = "" if anomaly_start_idx < 0 else str(times[anomaly_start_idx])
        fault_onset_time = "" if fault_onset_idx < 0 else str(times[fault_onset_idx])

        data: dict[str, object] = {
            "time": times.astype(str),
            "device_id": spec.device_id,
            "split": spec.split,
            "state_label": spec.state_label,
            "is_anomaly_point": is_anomaly.astype(int),
            "fault_onset_idx": fault_onset_idx,
            "fault_onset_time": fault_onset_time,
            "anomaly_start_idx": anomaly_start_idx,
            "anomaly_start_time": anomaly_start_time,
            "fault_cell": fault_cell,
            "current_a": current,
            "ambient_temp_c": ambient,
        }
        for idx, col in enumerate(TEMP_COLS):
            data[col] = temp[:, idx]
        for idx, col in enumerate(VOLT_COLS):
            data[col] = volt[:, idx]
        for idx, col in enumerate(SOC_COLS):
            data[col] = soc[:, idx]

        frames.append(pd.DataFrame(data))
        meta_rows.append(
            {
                "device_id": spec.device_id,
                "split": spec.split,
                "state_label": spec.state_label,
                "fault_onset_idx": fault_onset_idx,
                "fault_onset_time": fault_onset_time,
                "anomaly_start_idx": anomaly_start_idx,
                "anomaly_start_time": anomaly_start_time,
                "fault_cell": fault_cell,
                "capacity_ah_mean": float(np.mean(spec.capacity_ah)),
            }
        )

    readings = pd.concat(frames, ignore_index=True)
    device_meta = pd.DataFrame(meta_rows)
    return readings, device_meta
