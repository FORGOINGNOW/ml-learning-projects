from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_ts_anomaly.simulation import simulate_readings


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate lithium battery BMS time-series data.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readings, metadata = simulate_readings(config)
    readings_path = args.output_dir / "simulated_bms_readings.csv"
    metadata_path = args.output_dir / "device_metadata.csv"
    readings.to_csv(readings_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    print(f"Saved readings: {readings_path.resolve()} shape={readings.shape}")
    print(f"Saved metadata: {metadata_path.resolve()} shape={metadata.shape}")


if __name__ == "__main__":
    main()
