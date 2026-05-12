from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from battery_ts_anomaly.features import build_windows
from battery_ts_anomaly.model import LSTMForecaster
from battery_ts_anomaly.schema import FEATURE_COLUMNS, TARGET_COLUMNS
from battery_ts_anomaly.scoring import anomaly_scores


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device(config: dict) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if bool(config.get("require_cuda", True)):
        raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is False.")
    return torch.device("cpu")


def scale_windows(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    scaled = scaler.transform(flat).reshape(original_shape)
    return scaled.astype(np.float32)


def run_epoch(
    model: LSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    loss_fn = torch.nn.SmoothL1Loss()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
        total += float(loss.item()) * len(batch_x)
        count += len(batch_x)
    return total / max(count, 1)


def predict_scaled(model: LSTMForecaster, x_scaled: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_scaled)), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            preds.append(model(batch_x).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM normal-behavior forecaster.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = choose_device(config)
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(args.features, parse_dates=["time"], low_memory=False)
    train_df = df[(df["split"] == "train") & (df["state_label"] == "normal")].copy()
    windows = build_windows(
        train_df,
        seq_len=int(config["seq_len"]),
        forecast_horizon=int(config["forecast_horizon"]),
    )

    indices = np.arange(len(windows.x))
    train_idx, val_idx = train_test_split(indices, test_size=0.18, random_state=int(config["seed"]), shuffle=True)
    x_scaler = StandardScaler().fit(windows.x[train_idx].reshape(-1, len(FEATURE_COLUMNS)))
    y_scaler = StandardScaler().fit(windows.y[train_idx])

    x_train = scale_windows(windows.x[train_idx], x_scaler)
    x_val = scale_windows(windows.x[val_idx], x_scaler)
    y_train = y_scaler.transform(windows.y[train_idx]).astype(np.float32)
    y_val = y_scaler.transform(windows.y[val_idx]).astype(np.float32)

    batch_size = int(config["batch_size"])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = LSTMForecaster(
        input_dim=len(FEATURE_COLUMNS),
        output_dim=len(TARGET_COLUMNS),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "models").mkdir(parents=True, exist_ok=True)
    history_rows: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    model_path = args.run_dir / "models" / "lstm_forecaster_best.pt"

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr})
        print(f"epoch={epoch:02d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.6g}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "feature_columns": FEATURE_COLUMNS,
                    "target_columns": TARGET_COLUMNS,
                    "x_mean": x_scaler.mean_.astype(np.float32),
                    "x_scale": x_scaler.scale_.astype(np.float32),
                    "y_mean": y_scaler.mean_.astype(np.float32),
                    "y_scale": y_scaler.scale_.astype(np.float32),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                model_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stop_patience"]):
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    history = pd.DataFrame(history_rows)
    history_path = args.run_dir / "train_history.csv"
    history.to_csv(history_path, index=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    all_x_scaled = scale_windows(windows.x, x_scaler)
    pred_scaled = predict_scaled(model, all_x_scaled, device, batch_size=batch_size)
    y_pred = pred_scaled * y_scaler.scale_.reshape(1, -1) + y_scaler.mean_.reshape(1, -1)
    scores, residual_z = anomaly_scores(windows.y, y_pred, y_scaler.scale_)
    threshold = float(np.quantile(scores, float(config["threshold_quantile"])))

    threshold_path = args.run_dir / "threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "quantile": float(config["threshold_quantile"]),
                "train_score_mean": float(np.mean(scores)),
                "train_score_p95": float(np.quantile(scores, 0.95)),
                "train_score_p99": float(np.quantile(scores, 0.99)),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    residual_df = windows.meta.copy()
    residual_df["anomaly_score"] = scores
    for idx, col in enumerate(TARGET_COLUMNS):
        residual_df[f"abs_norm_residual_{col}"] = residual_z[:, idx]
    residual_path = args.run_dir / "train_residual_scores.csv"
    residual_df.to_csv(residual_path, index=False)

    checkpoint["threshold"] = threshold
    checkpoint["threshold_quantile"] = float(config["threshold_quantile"])
    torch.save(checkpoint, model_path)
    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved history: {history_path.resolve()}")
    print(f"Saved threshold: {threshold_path.resolve()} threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
