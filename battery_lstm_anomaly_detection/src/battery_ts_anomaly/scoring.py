from __future__ import annotations

import numpy as np

from battery_ts_anomaly.schema import TARGET_COLUMNS


TARGET_WEIGHTS = {
    "cell_voltage_min": 2.0,
    "cell_voltage_delta": 2.2,
    "cell_temp_max": 1.6,
    "cell_temp_delta": 1.5,
    "cell_soc_mean": 1.0,
    "cell_soc_delta": 1.3,
}


def target_weight_array() -> np.ndarray:
    return np.asarray([TARGET_WEIGHTS[col] for col in TARGET_COLUMNS], dtype=np.float32)


def residual_table(y_true: np.ndarray, y_pred: np.ndarray, target_scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(target_scale, dtype=np.float32).reshape(1, -1)
    return np.abs(y_true - y_pred) / np.maximum(scale, 1e-6)


def anomaly_scores(y_true: np.ndarray, y_pred: np.ndarray, target_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual_z = residual_table(y_true, y_pred, target_scale)
    weighted = residual_z * target_weight_array().reshape(1, -1)
    scores = 0.7 * np.max(weighted, axis=1) + 0.3 * np.mean(weighted, axis=1)
    return scores.astype(np.float32), residual_z.astype(np.float32)


def top_contributors(residual_z: np.ndarray) -> list[str]:
    weighted = residual_z * target_weight_array().reshape(1, -1)
    idx = np.argmax(weighted, axis=1)
    return [TARGET_COLUMNS[int(i)] for i in idx]
