from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from battery_ts_anomaly.features import build_windows
from battery_ts_anomaly.model import LSTMForecaster
from battery_ts_anomaly.schema import FEATURE_COLUMNS, TARGET_COLUMNS
from battery_ts_anomaly.scoring import anomaly_scores, top_contributors


def choose_device(config: dict) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if bool(config.get("require_cuda", True)):
        raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is False.")
    return torch.device("cpu")


def scale_windows(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / np.maximum(scale.reshape(1, 1, -1), 1e-6)).astype(np.float32)


def predict_scaled(model: LSTMForecaster, x_scaled: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_scaled)), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            preds.append(model(batch_x).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def metrics_for_group(group: pd.DataFrame, split: str) -> dict:
    y_true = group["is_anomaly_point"].astype(int).to_numpy()
    y_pred = group["is_detected"].astype(int).to_numpy()
    score = group["anomaly_score"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "split": split,
        "windows": int(len(group)),
        "positives": int(y_true.sum()),
        "threshold": float(group["threshold"].iloc[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, score),
        "average_precision": safe_ap(y_true, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_device_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    pred = predictions.copy()
    pred["target_time"] = pd.to_datetime(pred["target_time"])
    pred["anomaly_start_time_dt"] = pd.to_datetime(pred["anomaly_start_time"], errors="coerce")
    for (split, device_id), group in pred.groupby(["split", "device_id"], sort=False):
        state = str(group["state_label"].iloc[0])
        threshold = float(group["threshold"].iloc[0])
        true_anomaly = state != "normal"
        max_score = float(group["anomaly_score"].max())
        detected_any = bool(group["is_detected"].any())
        delay_minutes = float("nan")
        first_detect_time = ""
        if true_anomaly:
            start_time = group["anomaly_start_time_dt"].dropna().min()
            after_start = group[group["target_time"] >= start_time]
            detections = after_start[after_start["is_detected"]]
            if len(detections) > 0:
                first_time = detections["target_time"].min()
                first_detect_time = str(first_time)
                delay_minutes = float((first_time - start_time).total_seconds() / 60.0)
        rows.append(
            {
                "split": split,
                "device_id": device_id,
                "state_label": state,
                "true_anomaly": true_anomaly,
                "detected_any": detected_any,
                "max_score": max_score,
                "threshold": threshold,
                "first_detect_time": first_detect_time,
                "delay_minutes": delay_minutes,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LSTM residual anomaly detection.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("runs/models/lstm_forecaster_best.pt"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/model"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = choose_device(config)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = LSTMForecaster(
        input_dim=len(FEATURE_COLUMNS),
        output_dim=len(TARGET_COLUMNS),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    df = pd.read_csv(args.features, parse_dates=["time"], low_memory=False)
    windows = build_windows(
        df,
        seq_len=int(config["seq_len"]),
        forecast_horizon=int(config["forecast_horizon"]),
    )
    x_scaled = scale_windows(windows.x, checkpoint["x_mean"], checkpoint["x_scale"])
    pred_scaled = predict_scaled(model, x_scaled, device, int(config["batch_size"]))
    y_scale = np.asarray(checkpoint["y_scale"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_pred = pred_scaled * y_scale.reshape(1, -1) + y_mean.reshape(1, -1)
    scores, residual_z = anomaly_scores(windows.y, y_pred, y_scale)
    threshold = float(checkpoint["threshold"])

    pred_df = windows.meta.copy()
    pred_df["threshold"] = threshold
    pred_df["anomaly_score"] = scores
    pred_df["is_detected"] = pred_df["anomaly_score"] >= threshold
    pred_df["top_residual_metric"] = top_contributors(residual_z)
    for idx, col in enumerate(TARGET_COLUMNS):
        pred_df[f"actual_{col}"] = windows.y[:, idx]
        pred_df[f"pred_{col}"] = y_pred[:, idx]
        pred_df[f"residual_{col}"] = windows.y[:, idx] - y_pred[:, idx]
        pred_df[f"abs_norm_residual_{col}"] = residual_z[:, idx]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.report_dir / "window_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metric_rows = []
    for split, group in pred_df.groupby("split", sort=False):
        metric_rows.append(metrics_for_group(group, split))
    metrics = pd.DataFrame(metric_rows)
    metrics_path = args.report_dir / "metrics_by_split.csv"
    metrics.to_csv(metrics_path, index=False)

    state_rows = []
    eval_df = pred_df[pred_df["split"].isin(["valid", "test"])].copy()
    for (split, state), group in eval_df.groupby(["split", "state_label"], sort=False):
        y_true = group["is_anomaly_point"].astype(int).to_numpy()
        y_pred_bin = group["is_detected"].astype(int).to_numpy()
        state_rows.append(
            {
                "split": split,
                "state_label": state,
                "windows": int(len(group)),
                "positive_windows": int(y_true.sum()),
                "detected_windows": int(y_pred_bin.sum()),
                "detection_rate": float(y_pred_bin.mean()),
                "recall_if_abnormal": float(recall_score(y_true, y_pred_bin, zero_division=0)),
                "mean_score": float(group["anomaly_score"].mean()),
                "p95_score": float(group["anomaly_score"].quantile(0.95)),
            }
        )
    state_metrics = pd.DataFrame(state_rows)
    state_path = args.report_dir / "metrics_by_state.csv"
    state_metrics.to_csv(state_path, index=False)

    device_scores = build_device_scores(pred_df[pred_df["split"].isin(["valid", "test"])])
    device_path = args.report_dir / "device_scores.csv"
    device_scores.to_csv(device_path, index=False)

    summary_path = args.report_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "model_path": str(args.model),
                "prediction_rows": int(len(pred_df)),
                "metrics_by_split": metric_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved predictions: {pred_path.resolve()} shape={pred_df.shape}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
