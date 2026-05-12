from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full battery LSTM anomaly detection pipeline.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-devices", type=int, default=None)
    parser.add_argument("--valid-devices", type=int, default=None)
    parser.add_argument("--test-devices", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--threshold-quantile", type=float, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = {
        "epochs": args.epochs,
        "train_devices": args.train_devices,
        "valid_devices": args.valid_devices,
        "test_devices": args.test_devices,
        "days_per_device": args.days,
        "seq_len": args.seq_len,
        "threshold_quantile": args.threshold_quantile,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    runtime_path = root / "configs" / "_runtime.json"
    runtime_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    runtime_config = "configs/_runtime.json"
    py = sys.executable

    run([py, "src/simulate_data.py", "--config", runtime_config], root)
    run([py, "src/build_features.py", "--config", runtime_config], root)
    run([py, "src/visualize_data.py"], root)
    run([py, "src/train.py", "--config", runtime_config], root)
    run([py, "src/evaluate.py", "--config", runtime_config], root)
    run([py, "src/visualize_model.py"], root)
    run([py, "src/make_report.py", "--config", runtime_config], root)
    print("\nPipeline complete.")
    print("Report: reports/index.html")
    print("Model: runs/models/lstm_forecaster_best.pt")


if __name__ == "__main__":
    main()
