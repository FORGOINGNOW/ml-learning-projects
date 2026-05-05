"""数据生成脚本：根据配置生成多设备、多 cell、多状态的合成电池时序数据。"""

from __future__ import annotations

import argparse

from battery_classifier.simulation import save_simulation
from battery_classifier.utils import ensure_dirs, load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic battery multiclass data.")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    root = project_root()
    config = load_config(args.config)
    ensure_dirs(root)
    raw_path, metadata_path = save_simulation(config, root)
    print(f"Saved readings: {raw_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
