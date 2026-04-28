import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DrivingDataset
from model import EndToEndDrivingNet


TARGETS = ["steering", "throttle", "brake"]


class MetricLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(run_dir))
            self.mode = "tensorboard"
        except Exception:
            self.mode = "csv"
            self.csv_path = run_dir / "metrics.csv"
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "name", "value"])

    def scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
        else:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, name, value])

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights).mean()


def run_epoch(model, loader, device, weights, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_abs = torch.zeros(3, device=device)
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = weighted_mse(pred, y, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch = len(x)
        total_loss += loss.item() * batch
        total_abs += (pred.detach() - y).abs().sum(dim=0)
        count += batch
    mae = (total_abs / count).detach().cpu().numpy()
    return total_loss / count, mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an end-to-end visual driving model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/e2e_cnn"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    device = torch.device("cuda")

    train_ds = DrivingDataset(args.processed_dir / "train.csv", args.raw_dir, augment=True)
    val_ds = DrivingDataset(args.processed_dir / "val.csv", args.raw_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    model = EndToEndDrivingNet().to(device)
    weights = torch.tensor(
        [cfg["steering_loss_weight"], cfg["throttle_loss_weight"], cfg["brake_loss_weight"]],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    logger = MetricLogger(args.run_dir)

    best_val = float("inf")
    best_path = args.run_dir / "best_model.pt"
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss, train_mae = run_epoch(model, train_loader, device, weights, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, device, weights)
        scheduler.step(val_loss)
        logger.scalar("loss/train", train_loss, epoch)
        logger.scalar("loss/val", val_loss, epoch)
        for i, name in enumerate(TARGETS):
            logger.scalar(f"mae/train_{name}", float(train_mae[i]), epoch)
            logger.scalar(f"mae/val_{name}", float(val_mae[i]), epoch)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_mae steering={val_mae[0]:.4f} throttle={val_mae[1]:.4f} brake={val_mae[2]:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss}, best_path)

    logger.close()
    print(f"Saved best checkpoint to {best_path.resolve()}")
    print(f"Logging mode: {logger.mode}")


if __name__ == "__main__":
    main()
