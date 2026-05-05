"""预测脚本：读取真实或外部原始 BMS CSV，构造特征并输出状态分类结果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_classifier.features import build_feature_table
from battery_classifier.inference import aggregate_device_predictions, load_model_index, predict_feature_table
from battery_classifier.utils import ensure_dirs, load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict battery state labels for a raw BMS CSV.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--raw", required=True, help="Raw CSV with time/devices/cell/current columns.")
    parser.add_argument("--output-dir", default="reports/predictions")
    args = parser.parse_args()

    root = project_root()
    ensure_dirs(root)
    config = load_config(args.config)
    raw = pd.read_csv(args.raw)
    features = build_feature_table(raw, float(config["current_deadband_a"]))
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "features.csv", index=False)

    index = load_model_index(root)
    class_names = list(index["class_names"])
    sample_preds = predict_feature_table(features, root, split=None)
    device_preds = aggregate_device_predictions(sample_preds, class_names)
    sample_preds.to_csv(output_dir / "sample_predictions.csv", index=False)
    device_preds.to_csv(output_dir / "device_predictions.csv", index=False)
    print(f"Saved predictions: {output_dir}")


if __name__ == "__main__":
    main()
