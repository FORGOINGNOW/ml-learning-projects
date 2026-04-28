import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import DrivingDataset
from model import EndToEndDrivingNet


def normalize_map(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create saliency visualizations for the end-to-end driving model.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/e2e_cnn/best_model.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/explain"))
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for explanation.")
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = DrivingDataset(args.processed_dir / "test.csv", args.raw_dir, augment=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EndToEndDrivingNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pred_csv = args.out_dir.parent / "model" / "test_predictions.csv"
    if pred_csv.exists():
        pred_df = pd.read_csv(pred_csv).sort_values("abs_error_steering", ascending=False).head(args.num_samples)
        indices = pred_df.index.tolist()
    else:
        indices = np.linspace(0, len(ds) - 1, min(args.num_samples, len(ds)), dtype=int).tolist()

    rows = []
    for panel_idx, idx in enumerate(indices):
        x, y = ds[idx]
        inp = x.unsqueeze(0).to(device)
        inp.requires_grad_(True)
        pred = model(inp)
        model.zero_grad(set_to_none=True)
        pred[0, 0].backward()
        saliency = inp.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
        saliency = normalize_map(saliency)
        img = np.asarray(Image.open(args.raw_dir / ds.df.iloc[idx]["image_path"]).convert("RGB")).astype(np.float32) / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(img)
        axes[0].set_title("input")
        axes[1].imshow(saliency, cmap="inferno")
        axes[1].set_title("steering saliency")
        axes[2].imshow(img)
        axes[2].imshow(saliency, cmap="inferno", alpha=0.45)
        axes[2].set_title("overlay")
        for ax in axes:
            ax.axis("off")
        true_values = y.numpy()
        pred_values = pred.detach().cpu().numpy()[0]
        fig.suptitle(
            f"true steering={true_values[0]:+.3f}, pred={pred_values[0]:+.3f}; "
            f"throttle={pred_values[1]:.3f}, brake={pred_values[2]:.3f}",
            fontsize=10,
        )
        out_path = args.out_dir / f"saliency_{panel_idx:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        rows.append(
            {
                "image_path": ds.df.iloc[idx]["image_path"],
                "saliency_path": str(out_path.relative_to(args.out_dir.parent)),
                "true_steering": float(true_values[0]),
                "pred_steering": float(pred_values[0]),
                "pred_throttle": float(pred_values[1]),
                "pred_brake": float(pred_values[2]),
            }
        )

    pd.DataFrame(rows).to_csv(args.out_dir / "saliency_index.csv", index=False)
    print(f"Saved saliency visualizations to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
