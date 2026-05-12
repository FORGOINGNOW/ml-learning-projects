from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from battery_ts_anomaly.features import add_engineered_features, save_feature_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BMS pack-level time-series features.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/simulated_bms_readings.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw, low_memory=False)
    features = add_engineered_features(raw, config)
    feature_path = args.output_dir / "features.csv"
    manifest_path = args.output_dir / "feature_manifest.csv"
    features.to_csv(feature_path, index=False)
    save_feature_manifest(str(manifest_path), config)
    print(f"Saved features: {feature_path.resolve()} shape={features.shape}")
    print(f"Saved manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
