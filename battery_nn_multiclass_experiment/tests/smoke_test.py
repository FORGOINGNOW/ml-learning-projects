"""冒烟测试脚本：快速验证合成数据、字段约束和特征构造流程是否可用。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery_classifier.features import build_feature_table
from battery_classifier.schema import SOC_COLS, TEMP_COLS, VOLT_COLS
from battery_classifier.simulation import simulate_readings


def main() -> None:
    config = {
        "seed": 7,
        "rated_capacity_ah": 17.0,
        "sample_interval_minutes": 2,
        "days_per_device": 1,
        "n_cells": 8,
        "train_devices": 10,
        "valid_devices": 5,
        "test_devices": 5,
        "abnormal_states": [
            "outliers",
            "level_shift",
            "gradual_drift",
            "battery_replacement",
            "normal",
        ],
    }
    raw, metadata = simulate_readings(config)
    assert len(metadata) == 20
    assert set(TEMP_COLS + VOLT_COLS + SOC_COLS).issubset(raw.columns)
    assert raw[TEMP_COLS].min().min() >= 20.0
    assert raw[TEMP_COLS].max().max() <= 60.0
    assert raw[VOLT_COLS].max().max() <= 3.5

    features = build_feature_table(raw, current_deadband=0.05)
    assert not features.empty
    assert {"charge", "discharge"}.issubset(set(features["condition"]))
    assert features["state_label"].nunique() == 5
    print("smoke test passed")


if __name__ == "__main__":
    main()
