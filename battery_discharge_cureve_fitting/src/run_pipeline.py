import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run battery discharge curve fitting pipeline.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    config = "configs/default.json"
    if args.epochs is not None:
        cfg = json.loads((root / config).read_text(encoding="utf-8"))
        cfg["epochs"] = args.epochs
        runtime = root / "configs" / "_runtime.json"
        runtime.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        config = "configs/_runtime.json"

    run([py, "src/simulate_data.py", "--config", config], root)
    run([py, "src/visualize_data.py"], root)
    run([py, "src/train.py", "--config", config], root)
    run([py, "src/evaluate.py", "--config", config], root)
    run([py, "src/make_report.py"], root)
    print("\nPipeline complete.")
    print("Report: reports/index.html")
    print("Training logs: runs/fit_logs")


if __name__ == "__main__":
    main()
