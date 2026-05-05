"""特征构造脚本：把原始 BMS 时序数据聚合成设备-日期-工况级训练特征。"""

from __future__ import annotations

import argparse

from battery_classifier.features import save_feature_table
from battery_classifier.utils import ensure_dirs, load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build device-day-condition features.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--raw", default="data/raw/simulated_battery_readings.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()

    root = project_root()
    config = load_config(args.config)
    ensure_dirs(root)
    out = save_feature_table(
        raw_path=root / args.raw,
        output_path=root / args.output,
        current_deadband=float(config["current_deadband_a"]),
    )
    print(f"Saved features: {out}")


if __name__ == "__main__":
    main()
