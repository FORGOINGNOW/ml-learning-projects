import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


TARGETS = ["steering", "throttle", "brake"]


def inspect_image(path: Path, width: int, height: int) -> dict:
    try:
        with Image.open(path) as img:
            ok_size = img.size == (width, height)
            arr = np.asarray(img.convert("RGB"))
        brightness = float(arr.mean())
        contrast = float(arr.std())
        valid = ok_size and 15.0 <= brightness <= 245.0 and contrast >= 8.0
        return {"exists": 1, "ok_size": int(ok_size), "brightness": brightness, "contrast": contrast, "valid_image": int(valid)}
    except Exception:
        return {"exists": 0, "ok_size": 0, "brightness": np.nan, "contrast": np.nan, "valid_image": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean synthetic driving dataset and create train/val/test manifests.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    width, height = int(cfg["image_width"]), int(cfg["image_height"])
    df = pd.read_csv(args.raw_dir / "metadata.csv")

    checks = [inspect_image(args.raw_dir / p, width, height) for p in df["image_path"]]
    checked = pd.concat([df, pd.DataFrame(checks)], axis=1)

    label_ok = np.ones(len(checked), dtype=bool)
    label_ok &= checked["steering"].between(-1, 1)
    label_ok &= checked["throttle"].between(0, 1)
    label_ok &= checked["brake"].between(0, 1)
    checked["valid_label"] = label_ok.astype(int)
    checked["keep"] = ((checked["valid_image"] == 1) & (checked["valid_label"] == 1)).astype(int)

    clean = checked[checked["keep"] == 1].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checked.to_csv(args.out_dir / "quality_report.csv", index=False)
    clean.to_csv(args.out_dir / "manifest_all.csv", index=False)
    for split in ["train", "val", "test"]:
        clean[clean["split_hint"] == split].to_csv(args.out_dir / f"{split}.csv", index=False)

    print(f"Raw rows: {len(df)}")
    print(f"Kept rows: {len(clean)}")
    print("Removed rows:", int((checked["keep"] == 0).sum()))
    print(clean["split_hint"].value_counts().to_string())


if __name__ == "__main__":
    main()
