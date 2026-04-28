import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from features import FEATURE_COLUMNS, add_features, make_curve_arrays, make_point_arrays
from models import CNN1DRegressor, DNNRegressor


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
        if self.writer:
            self.writer.add_scalar(name, value, step)
        else:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, name, value])

    def close(self) -> None:
        if self.writer:
            self.writer.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (((pred - y) ** 2) * weights).mean()


def run_point_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for x, y, weights in loader:
        x, y, weights = x.to(device), y.to(device), weights.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = weighted_mse(pred, y, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def run_curve_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for x, y, weights in loader:
        x, y, weights = x.to(device), y.to(device), weights.to(device)
        x = x.transpose(1, 2)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = weighted_mse(pred, y, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DNN and 1D-CNN discharge voltage fitting models.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--data", type=Path, default=Path("data/discharge_df.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural network training.")
    device = torch.device("cuda")
    df = pd.read_csv(args.data)
    rated_capacity = float(cfg["rated_capacity_ah"])

    x, y = make_point_arrays(df, rated_capacity)
    point_features = add_features(df, rated_capacity)
    point_weights = (1.0 + float(cfg.get("tail_loss_boost", 5.0)) * point_features["tail_progress"].to_numpy(dtype=np.float32)).reshape(-1, 1)
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x, y, point_weights, test_size=0.18, random_state=cfg["seed"]
    )
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)
    y_train_s = y_scaler.transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    dnn = DNNRegressor(len(FEATURE_COLUMNS)).to(device)
    dnn_opt = torch.optim.AdamW(dnn.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    dnn_train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_s), torch.tensor(y_train_s), torch.tensor(w_train.astype(np.float32))),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    dnn_val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val_s), torch.tensor(y_val_s), torch.tensor(w_val.astype(np.float32))),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    logger = MetricLogger(args.run_dir / "fit_logs")
    best_dnn = float("inf")
    dnn_path = args.run_dir / "dnn_best.pt"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(cfg["epochs"]) + 1):
        tr = run_point_epoch(dnn, dnn_train_loader, device, dnn_opt)
        va = run_point_epoch(dnn, dnn_val_loader, device)
        logger.scalar("dnn/train_mse_scaled", tr, epoch)
        logger.scalar("dnn/val_mse_scaled", va, epoch)
        if va < best_dnn:
            best_dnn = va
            torch.save({"model": dnn.state_dict(), "x_mean": x_scaler.mean_, "x_scale": x_scaler.scale_, "y_mean": y_scaler.mean_, "y_scale": y_scaler.scale_, "features": FEATURE_COLUMNS, "config": cfg}, dnn_path)
        print(f"DNN epoch={epoch:02d} train={tr:.6f} val={va:.6f}")

    curves_x, curves_y, _ = make_curve_arrays(df, rated_capacity)
    tail_idx = FEATURE_COLUMNS.index("tail_progress")
    curve_weights = (1.0 + float(cfg.get("tail_loss_boost", 5.0)) * curves_x[:, :, tail_idx : tail_idx + 1]).astype(np.float32)
    curve_train_idx, curve_val_idx = train_test_split(np.arange(len(curves_x)), test_size=0.18, random_state=cfg["seed"])
    flat_train = curves_x[curve_train_idx].reshape(-1, curves_x.shape[-1])
    flat_y_train = curves_y[curve_train_idx].reshape(-1, 1)
    cx_scaler = StandardScaler().fit(flat_train)
    cy_scaler = StandardScaler().fit(flat_y_train)
    curves_x_s = cx_scaler.transform(curves_x.reshape(-1, curves_x.shape[-1])).reshape(curves_x.shape).astype(np.float32)
    curves_y_s = cy_scaler.transform(curves_y.reshape(-1, 1)).reshape(curves_y.shape).astype(np.float32)

    cnn = CNN1DRegressor(len(FEATURE_COLUMNS)).to(device)
    cnn_opt = torch.optim.AdamW(cnn.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    cnn_train_loader = DataLoader(
        TensorDataset(torch.tensor(curves_x_s[curve_train_idx]), torch.tensor(curves_y_s[curve_train_idx]), torch.tensor(curve_weights[curve_train_idx])),
        batch_size=32,
        shuffle=True,
    )
    cnn_val_loader = DataLoader(
        TensorDataset(torch.tensor(curves_x_s[curve_val_idx]), torch.tensor(curves_y_s[curve_val_idx]), torch.tensor(curve_weights[curve_val_idx])),
        batch_size=32,
        shuffle=False,
    )
    best_cnn = float("inf")
    cnn_path = args.run_dir / "cnn1d_best.pt"
    for epoch in range(1, int(cfg["epochs"]) + 1):
        tr = run_curve_epoch(cnn, cnn_train_loader, device, cnn_opt)
        va = run_curve_epoch(cnn, cnn_val_loader, device)
        logger.scalar("cnn1d/train_mse_scaled", tr, epoch)
        logger.scalar("cnn1d/val_mse_scaled", va, epoch)
        if va < best_cnn:
            best_cnn = va
            torch.save({"model": cnn.state_dict(), "x_mean": cx_scaler.mean_, "x_scale": cx_scaler.scale_, "y_mean": cy_scaler.mean_, "y_scale": cy_scaler.scale_, "features": FEATURE_COLUMNS, "config": cfg}, cnn_path)
        print(f"1D-CNN epoch={epoch:02d} train={tr:.6f} val={va:.6f}")

    logger.close()
    print(f"Saved DNN checkpoint: {dnn_path.resolve()}")
    print(f"Saved 1D-CNN checkpoint: {cnn_path.resolve()}")
    print(f"Logging mode: {logger.mode}")


if __name__ == "__main__":
    main()
