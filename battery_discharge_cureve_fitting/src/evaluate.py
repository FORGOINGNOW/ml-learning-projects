import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import FEATURE_COLUMNS, add_features, make_curve_arrays, make_point_arrays
from models import CNN1DRegressor, DNNRegressor


def scale(x: np.ndarray, mean: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    return ((x - mean) / scale_).astype(np.float32)


def unscale(y: np.ndarray, mean: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    return y * scale_ + mean


def predict_dnn(df: pd.DataFrame, checkpoint: dict, rated_capacity: float, device) -> np.ndarray:
    x, _ = make_point_arrays(df, rated_capacity)
    model = DNNRegressor(len(FEATURE_COLUMNS)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    xs = scale(x, checkpoint["x_mean"], checkpoint["x_scale"])
    preds = []
    with torch.no_grad():
        for start in range(0, len(xs), 4096):
            batch = torch.tensor(xs[start : start + 4096], dtype=torch.float32, device=device)
            preds.append(model(batch).detach().cpu().numpy())
    pred_scaled = np.vstack(preds)
    return unscale(pred_scaled, checkpoint["y_mean"], checkpoint["y_scale"]).reshape(-1)


def predict_cnn(df: pd.DataFrame, checkpoint: dict, rated_capacity: float, device) -> np.ndarray:
    x, _, _ = make_curve_arrays(df, rated_capacity)
    model = CNN1DRegressor(len(FEATURE_COLUMNS)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    xs = scale(x.reshape(-1, x.shape[-1]), checkpoint["x_mean"], checkpoint["x_scale"]).reshape(x.shape)
    preds = []
    with torch.no_grad():
        for start in range(0, len(xs), 32):
            batch = torch.tensor(xs[start : start + 32], dtype=torch.float32, device=device).transpose(1, 2)
            preds.append(model(batch).detach().cpu().numpy())
    pred_scaled = np.vstack(preds).reshape(-1, 1)
    return unscale(pred_scaled, checkpoint["y_mean"], checkpoint["y_scale"]).reshape(-1)


def infer_curve_ids(df: pd.DataFrame) -> np.ndarray:
    ids = np.zeros(len(df), dtype=np.int64)
    current = 0
    prev_c = None
    prev_t = -np.inf
    for i, (c, t) in enumerate(zip(df["放电倍率"].to_numpy(), df["放电时间"].to_numpy())):
        if i > 0 and (c != prev_c or t < prev_t):
            current += 1
        ids[i] = current
        prev_c, prev_t = c, t
    return ids


def metrics(y_true, y_pred, name: str, tail_mask: np.ndarray) -> dict:
    return {
        "model": name,
        "mae_v": mean_absolute_error(y_true, y_pred),
        "rmse_v": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "tail_mae_v": mean_absolute_error(y_true[tail_mask], y_pred[tail_mask]),
        "tail_rmse_v": float(np.sqrt(mean_squared_error(y_true[tail_mask], y_pred[tail_mask]))),
        "r2": r2_score(y_true, y_pred),
        "max_abs_error_v": float(np.max(np.abs(y_true - y_pred))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DNN and 1D-CNN on validation discharge curves.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--valid", type=Path, default=Path("data/valid_df.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/model"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rated_capacity = float(cfg["rated_capacity_ah"])
    valid_df = pd.read_csv(args.valid)
    y_true = valid_df["放电电压"].to_numpy(dtype=np.float32)
    feature_df = add_features(valid_df, rated_capacity)
    tail_mask = feature_df["curve_capacity_frac"].to_numpy() >= 0.8

    dnn_ckpt = torch.load(args.run_dir / "dnn_best.pt", map_location=device, weights_only=False)
    cnn_ckpt = torch.load(args.run_dir / "cnn1d_best.pt", map_location=device, weights_only=False)
    valid_df["dnn_pred_voltage"] = predict_dnn(valid_df, dnn_ckpt, rated_capacity, device)
    valid_df["cnn1d_pred_voltage"] = predict_cnn(valid_df, cnn_ckpt, rated_capacity, device)
    valid_df["curve_id"] = infer_curve_ids(valid_df)

    metrics_df = pd.DataFrame(
        [
            metrics(y_true, valid_df["dnn_pred_voltage"].to_numpy(), "DNN", tail_mask),
            metrics(y_true, valid_df["cnn1d_pred_voltage"].to_numpy(), "1D-CNN", tail_mask),
        ]
    ).sort_values("rmse_v")
    metrics_df.to_csv(args.out_dir / "valid_metrics.csv", index=False)
    valid_df.to_csv(args.out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    plot_curves = valid_df[valid_df["curve_id"].isin(valid_df.groupby("放电倍率")["curve_id"].min().values)]
    for c_rate, part in plot_curves.groupby("放电倍率"):
        plt.figure(figsize=(9, 5))
        plt.plot(part["放电时间"], part["放电电压"], label="true degraded/valid", linewidth=2)
        plt.plot(part["放电时间"], part["dnn_pred_voltage"], label="DNN", linestyle="--")
        plt.plot(part["放电时间"], part["cnn1d_pred_voltage"], label="1D-CNN", linestyle=":")
        plt.title(f"Validation Curve Fitting at {c_rate}C")
        plt.xlabel("Discharge time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / f"curve_fit_{c_rate}C.png", dpi=170)
        plt.close()

    plot_df = valid_df.rename(columns={"放电倍率": "c_rate", "放电电压": "voltage_v"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pred_col, title in zip(axes, ["dnn_pred_voltage", "cnn1d_pred_voltage"], ["DNN", "1D-CNN"]):
        sns.scatterplot(data=plot_df.sample(min(len(plot_df), 2500), random_state=42), x="voltage_v", y=pred_col, hue="c_rate", palette="viridis", s=14, ax=ax)
        lo = min(plot_df["voltage_v"].min(), plot_df[pred_col].min())
        hi = max(plot_df["voltage_v"].max(), plot_df[pred_col].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{title}: Predicted vs True")
        ax.set_xlabel("True voltage (V)")
        ax.set_ylabel("Predicted voltage (V)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "prediction_scatter.png", dpi=170)
    plt.close()

    err_df = plot_df[["c_rate"]].copy()
    err_df["DNN_abs_error"] = (valid_df["dnn_pred_voltage"] - valid_df["放电电压"]).abs()
    err_df["1D-CNN_abs_error"] = (valid_df["cnn1d_pred_voltage"] - valid_df["放电电压"]).abs()
    err_long = err_df.melt(id_vars="c_rate", var_name="model", value_name="abs_error_v")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=err_long, x="c_rate", y="abs_error_v", hue="model")
    plt.title("Validation Absolute Error by C-rate")
    plt.ylabel("Absolute error (V)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "abs_error_by_c_rate.png", dpi=170)
    plt.close()

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"best_model": metrics_df.iloc[0]["model"], "metrics": metrics_df.to_dict(orient="records")}, f, ensure_ascii=False, indent=2)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved evaluation reports to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
