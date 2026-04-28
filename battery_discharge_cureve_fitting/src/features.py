import numpy as np
import pandas as pd


RATED_CAPACITY_AH = 17.0
FEATURE_COLUMNS = [
    "time_norm_by_c",
    "curve_time_frac",
    "c_rate",
    "capacity_norm",
    "curve_capacity_frac",
    "soc_est",
    "curve_soc_est",
    "soh_proxy",
    "current_a",
    "c_rate_sq",
    "sqrt_curve_time_frac",
    "tail_progress",
]


def infer_curve_ids(df: pd.DataFrame) -> np.ndarray:
    curve_ids = np.zeros(len(df), dtype=np.int64)
    current = 0
    prev_c = None
    prev_t = -np.inf
    for i, (c_rate, t) in enumerate(zip(df["放电倍率"].to_numpy(), df["放电时间"].to_numpy())):
        if i > 0 and (c_rate != prev_c or t < prev_t):
            current += 1
        curve_ids[i] = current
        prev_c = c_rate
        prev_t = t
    return curve_ids


def add_features(df: pd.DataFrame, rated_capacity_ah: float = RATED_CAPACITY_AH) -> pd.DataFrame:
    out = df.copy()
    curve_ids = infer_curve_ids(out)
    out["_curve_id"] = curve_ids
    max_time_by_c = out.groupby("放电倍率")["放电时间"].transform("max").replace(0, np.nan)
    max_time_by_curve = out.groupby("_curve_id")["放电时间"].transform("max").replace(0, np.nan)
    max_capacity_by_curve = out.groupby("_curve_id")["已放电容量"].transform("max").replace(0, np.nan)
    out["time_norm_by_c"] = (out["放电时间"] / max_time_by_c).fillna(0)
    out["curve_time_frac"] = (out["放电时间"] / max_time_by_curve).fillna(0)
    out["c_rate"] = out["放电倍率"]
    out["capacity_norm"] = out["已放电容量"] / rated_capacity_ah
    out["curve_capacity_frac"] = (out["已放电容量"] / max_capacity_by_curve).fillna(0)
    out["soc_est"] = 1.0 - out["capacity_norm"]
    out["curve_soc_est"] = 1.0 - out["curve_capacity_frac"]
    out["soh_proxy"] = (max_capacity_by_curve / rated_capacity_ah).fillna(1.0)
    out["current_a"] = out["放电倍率"] * rated_capacity_ah
    out["c_rate_sq"] = out["放电倍率"] ** 2
    out["sqrt_curve_time_frac"] = np.sqrt(np.clip(out["curve_time_frac"], 0, 1))
    out["tail_progress"] = np.clip((out["curve_capacity_frac"] - 0.78) / 0.22, 0, 1)
    return out


def make_point_arrays(df: pd.DataFrame, rated_capacity_ah: float = RATED_CAPACITY_AH):
    feat = add_features(df, rated_capacity_ah)
    x = feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = feat["放电电压"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y


def make_curve_arrays(df: pd.DataFrame, rated_capacity_ah: float = RATED_CAPACITY_AH):
    feat = add_features(df, rated_capacity_ah)
    curve_ids = infer_curve_ids(feat)
    xs, ys, meta = [], [], []
    for curve_id in np.unique(curve_ids):
        part = feat[curve_ids == curve_id]
        xs.append(part[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        ys.append(part["放电电压"].to_numpy(dtype=np.float32).reshape(-1, 1))
        meta.append({"curve_id": int(curve_id), "放电倍率": float(part["放电倍率"].iloc[0]), "points": int(len(part))})
    return np.stack(xs), np.stack(ys), pd.DataFrame(meta)
