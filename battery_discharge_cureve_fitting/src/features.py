import numpy as np
import pandas as pd


TIME_COL = "放电时间"
C_RATE_COL = "放电倍率"
VOLTAGE_COL = "放电电压"
CAPACITY_COL = "已放电容量"

RATED_CAPACITY_AH = 17.0
FEATURE_COLUMNS = [
    "rated_time_frac",
    "c_rate",
    "capacity_norm",
    "soc_est",
    "current_a",
    "c_rate_sq",
    "sqrt_capacity_norm",
    "tail_progress",
]


def infer_curve_ids(df: pd.DataFrame) -> np.ndarray:
    curve_ids = np.zeros(len(df), dtype=np.int64)
    current = 0
    prev_c = None
    prev_t = -np.inf
    for i, (c_rate, t) in enumerate(zip(df[C_RATE_COL].to_numpy(), df[TIME_COL].to_numpy())):
        if i > 0 and (c_rate != prev_c or t < prev_t):
            current += 1
        curve_ids[i] = current
        prev_c = c_rate
        prev_t = t
    return curve_ids


def add_features(df: pd.DataFrame, rated_capacity_ah: float = RATED_CAPACITY_AH) -> pd.DataFrame:
    """Build model features from the rated-capacity view only.

    The model is intended to learn normal battery behavior. It must not receive
    curve-level max capacity or SOH-derived features at validation time, because
    those would reveal the degradation we are trying to detect.
    """
    out = df.copy()
    out["_curve_id"] = infer_curve_ids(out)
    rated_duration_s = 3600.0 / out[C_RATE_COL].replace(0, np.nan)
    out["rated_time_frac"] = (out[TIME_COL] / rated_duration_s).fillna(0)
    out["c_rate"] = out[C_RATE_COL]
    out["capacity_norm"] = out[CAPACITY_COL] / rated_capacity_ah
    out["soc_est"] = 1.0 - out["capacity_norm"]
    out["current_a"] = out[C_RATE_COL] * rated_capacity_ah
    out["c_rate_sq"] = out[C_RATE_COL] ** 2
    out["sqrt_capacity_norm"] = np.sqrt(np.clip(out["capacity_norm"], 0, 1.2))
    out["tail_progress"] = np.clip((out["capacity_norm"] - 0.78) / 0.22, 0, 1)
    return out


def make_point_arrays(df: pd.DataFrame, rated_capacity_ah: float = RATED_CAPACITY_AH):
    feat = add_features(df, rated_capacity_ah)
    x = feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = feat[VOLTAGE_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y


def make_curve_arrays(df: pd.DataFrame, rated_capacity_ah: float = RATED_CAPACITY_AH):
    feat = add_features(df, rated_capacity_ah)
    curve_ids = infer_curve_ids(feat)
    xs, ys, meta = [], [], []
    for curve_id in np.unique(curve_ids):
        part = feat[curve_ids == curve_id]
        xs.append(part[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        ys.append(part[VOLTAGE_COL].to_numpy(dtype=np.float32).reshape(-1, 1))
        meta.append({"curve_id": int(curve_id), C_RATE_COL: float(part[C_RATE_COL].iloc[0]), "points": int(len(part))})
    return np.stack(xs), np.stack(ys), pd.DataFrame(meta)
