import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full synthetic driving pipeline.")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    run([py, "src/simulate_data.py", "--num-samples", str(args.samples)], root)
    run([py, "src/clean_dataset.py"], root)
    run([py, "src/visualize_data.py"], root)
    if args.epochs is None:
        run([py, "src/train.py"], root)
    else:
        cfg_path = root / "configs" / "default.json"
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["epochs"] = args.epochs
        tmp_cfg = root / "configs" / "_runtime.json"
        tmp_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        run([py, "src/train.py", "--config", "configs/_runtime.json"], root)
    run([py, "src/evaluate.py"], root)
    run([py, "src/explain_model.py"], root)
    run([py, "src/make_report.py"], root)
    print("\nPipeline complete.")
    print("Data report: reports/data")
    print("Model report: reports/model")
    print("Full report: reports/index.html")
    print("Training logs: runs/e2e_cnn")


if __name__ == "__main__":
    main()
