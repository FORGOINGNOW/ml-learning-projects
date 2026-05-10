import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


COLS = ["放电时间", "放电倍率", "放电电压", "已放电容量"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    soc = np.clip(soc, 0, 1)
    plateau = 3.08 + 1.05 / (1 + np.exp(-10.0 * (soc - 0.08)))
    high_soc_relax = 0.08 * np.tanh((soc - 0.85) * 7.0)
    end_drop = 0.82 * np.exp(-soc / 0.045)
    return plateau + high_soc_relax - end_drop


def generate_curve(c_rate: float, rated_capacity_ah: float, points: int, capacity_factor: float, rng: np.random.Generator) -> pd.DataFrame:
    actual_capacity = rated_capacity_ah * capacity_factor
    duration_h = actual_capacity / (rated_capacity_ah * c_rate)
    t = np.linspace(0, duration_h * 3600, points)
    discharged_capacity = np.linspace(0, actual_capacity, points)
    soc = 1.0 - discharged_capacity / actual_capacity

    internal_resistance = rng.normal(0.018, 0.0025)
    polarization = 0.020 * np.log1p(c_rate) + 0.035 * c_rate * (1 - soc) ** 2
    ohmic_drop = internal_resistance * c_rate * rated_capacity_ah
    thermal_sag = 0.018 * (c_rate ** 1.35) * np.sqrt(np.linspace(0, 1, points))
    voltage = ocv_from_soc(soc) - ohmic_drop - polarization - thermal_sag
    voltage += rng.normal(0, 0.006 + 0.0015 * c_rate, size=points)
    voltage = np.clip(voltage, 2.45, 4.25)

    return pd.DataFrame(
        {
            "放电时间": t,
            "放电倍率": c_rate,
            "放电电压": voltage,
            "已放电容量": discharged_capacity,
        },
        columns=COLS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic lithium battery discharge curves.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_parts = []
    train_meta = []
    for c_rate in cfg["c_rates"]:
        for curve_idx in range(int(cfg["train_curves_per_c"])):
            cap_factor = float(np.clip(rng.normal(1.0, 0.012), 0.97, 1.03))
            train_parts.append(generate_curve(c_rate, cfg["rated_capacity_ah"], cfg["points_per_curve"], cap_factor, rng))
            train_meta.append(
                {
                    "curve_id": len(train_meta),
                    "放电倍率": c_rate,
                    "capacity_factor": cap_factor,
                    "is_degraded": 0,
                    "split": "train",
                }
            )
    discharge_df = pd.concat(train_parts, ignore_index=True)

    valid_parts = []
    valid_meta = []
    for c_rate in cfg["c_rates"]:
        for curve_idx in range(int(cfg["valid_curves_per_c"])):
            if curve_idx == 0:
                cap_factor = float(np.clip(rng.normal(0.82, 0.012), 0.78, 0.86))
                is_degraded = 1
            else:
                cap_factor = float(np.clip(rng.normal(1.0, 0.015), 0.96, 1.03))
                is_degraded = 0
            valid_parts.append(generate_curve(c_rate, cfg["rated_capacity_ah"], cfg["points_per_curve"], cap_factor, rng))
            valid_meta.append(
                {
                    "curve_id": len(valid_meta),
                    "放电倍率": c_rate,
                    "capacity_factor": cap_factor,
                    "is_degraded": is_degraded,
                    "split": "valid",
                }
            )
    valid_df = pd.concat(valid_parts, ignore_index=True)

    discharge_df.to_csv(out_dir / "discharge_df.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(out_dir / "valid_df.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(train_meta).to_csv(out_dir / "train_curve_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(valid_meta).to_csv(out_dir / "valid_curve_meta.csv", index=False, encoding="utf-8-sig")
    print(f"Saved discharge_df: {(out_dir / 'discharge_df.csv').resolve()} rows={len(discharge_df)}")
    print(f"Saved valid_df: {(out_dir / 'valid_df.csv').resolve()} rows={len(valid_df)}")
    print(discharge_df.groupby("放电倍率")["放电电压"].agg(["min", "mean", "max"]).to_string())


if __name__ == "__main__":
    main()
