import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from features import (
    CAPACITY_COL,
    C_RATE_COL,
    FEATURE_COLUMNS,
    TIME_COL,
    VOLTAGE_COL,
    add_features,
    infer_curve_ids,
    make_curve_arrays,
    make_point_arrays,
)
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
    return unscale(np.vstack(preds), checkpoint["y_mean"], checkpoint["y_scale"]).reshape(-1)


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
    return unscale(np.vstack(preds).reshape(-1, 1), checkpoint["y_mean"], checkpoint["y_scale"]).reshape(-1)


def regression_metrics(y_true, y_pred, name: str, mask: np.ndarray) -> dict:
    return {
        "model": name,
        "subset": "normal_validation",
        "mae_v": mean_absolute_error(y_true[mask], y_pred[mask]),
        "rmse_v": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "r2": r2_score(y_true[mask], y_pred[mask]),
        "max_abs_error_v": float(np.max(np.abs(y_true[mask] - y_pred[mask]))),
    }


def attach_curve_meta(valid_df: pd.DataFrame, meta_path: Path) -> pd.DataFrame:
    out = valid_df.copy()
    out["curve_id"] = infer_curve_ids(out)
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        out = out.merge(meta[["curve_id", "capacity_factor", "is_degraded"]], on="curve_id", how="left")
    else:
        out["capacity_factor"] = out.groupby("curve_id")[CAPACITY_COL].transform("max") / out[CAPACITY_COL].max()
        out["is_degraded"] = (out["capacity_factor"] < 0.9).astype(int)
    out["is_degraded"] = out["is_degraded"].fillna(0).astype(int)
    return out


def build_curve_scores(df: pd.DataFrame, pred_col: str, rated_capacity: float) -> pd.DataFrame:
    feat = add_features(df, rated_capacity)
    scored = df.copy()
    scored["capacity_norm"] = feat["capacity_norm"]
    scored["positive_residual_v"] = np.clip(scored[pred_col] - scored[VOLTAGE_COL], 0, None)
    rows = []
    for curve_id, part in scored.groupby("curve_id"):
        tail = part[part["capacity_norm"] >= 0.75]
        target = tail if len(tail) else part
        rows.append(
            {
                "curve_id": int(curve_id),
                C_RATE_COL: float(part[C_RATE_COL].iloc[0]),
                "capacity_factor": float(part["capacity_factor"].iloc[0]),
                "is_degraded": int(part["is_degraded"].iloc[0]),
                "max_capacity_ah": float(part[CAPACITY_COL].max()),
                "tail_positive_residual_v": float(target["positive_residual_v"].mean()),
                "tail_abs_error_v": float((target[pred_col] - target[VOLTAGE_COL]).abs().mean()),
                "end_positive_residual_v": float(part["positive_residual_v"].tail(max(5, len(part) // 20)).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normal-model voltage fitting and degraded-battery residuals.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--valid", type=Path, default=Path("data/valid_df.csv"))
    parser.add_argument("--valid-meta", type=Path, default=Path("data/valid_curve_meta.csv"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/model"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rated_capacity = float(cfg["rated_capacity_ah"])
    valid_df = attach_curve_meta(pd.read_csv(args.valid), args.valid_meta)
    y_true = valid_df[VOLTAGE_COL].to_numpy(dtype=np.float32)

    dnn_ckpt = torch.load(args.run_dir / "dnn_best.pt", map_location=device, weights_only=False)
    cnn_ckpt = torch.load(args.run_dir / "cnn1d_best.pt", map_location=device, weights_only=False)
    valid_df["dnn_pred_voltage"] = predict_dnn(valid_df, dnn_ckpt, rated_capacity, device)
    valid_df["cnn1d_pred_voltage"] = predict_cnn(valid_df, cnn_ckpt, rated_capacity, device)
    valid_df["dnn_positive_residual_v"] = np.clip(valid_df["dnn_pred_voltage"] - valid_df[VOLTAGE_COL], 0, None)
    valid_df["cnn1d_positive_residual_v"] = np.clip(valid_df["cnn1d_pred_voltage"] - valid_df[VOLTAGE_COL], 0, None)

    normal_mask = valid_df["is_degraded"].to_numpy() == 0
    metrics_df = pd.DataFrame(
        [
            regression_metrics(y_true, valid_df["dnn_pred_voltage"].to_numpy(), "DNN", normal_mask),
            regression_metrics(y_true, valid_df["cnn1d_pred_voltage"].to_numpy(), "1D-CNN", normal_mask),
        ]
    ).sort_values("rmse_v")

    dnn_scores = build_curve_scores(valid_df, "dnn_pred_voltage", rated_capacity)
    dnn_scores["model"] = "DNN"
    cnn_scores = build_curve_scores(valid_df, "cnn1d_pred_voltage", rated_capacity)
    cnn_scores["model"] = "1D-CNN"
    curve_scores = pd.concat([dnn_scores, cnn_scores], ignore_index=True)
    detection_rows = []
    for model_name, part in curve_scores.groupby("model"):
        if part["is_degraded"].nunique() > 1:
            auc = roc_auc_score(part["is_degraded"], part["tail_positive_residual_v"])
        else:
            auc = np.nan
        detection_rows.append(
            {
                "model": model_name,
                "degradation_score": "tail_positive_residual_v",
                "curve_roc_auc": auc,
                "normal_mean_score": part.loc[part["is_degraded"] == 0, "tail_positive_residual_v"].mean(),
                "degraded_mean_score": part.loc[part["is_degraded"] == 1, "tail_positive_residual_v"].mean(),
            }
        )
    detection_df = pd.DataFrame(detection_rows).sort_values("curve_roc_auc", ascending=False)

    metrics_df.to_csv(args.out_dir / "valid_metrics_normal_only.csv", index=False)
    detection_df.to_csv(args.out_dir / "degradation_detection_metrics.csv", index=False)
    curve_scores.to_csv(args.out_dir / "curve_degradation_scores.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(args.out_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    degraded_curve_ids = valid_df.loc[valid_df["is_degraded"] == 1, "curve_id"].unique()
    plot_curves = valid_df[valid_df["curve_id"].isin(degraded_curve_ids)]
    for c_rate, part in plot_curves.groupby(C_RATE_COL):
        plt.figure(figsize=(9, 5))
        plt.plot(part[TIME_COL], part[VOLTAGE_COL], label="actual degraded curve", linewidth=2)
        plt.plot(part[TIME_COL], part["dnn_pred_voltage"], label="normal model: DNN", linestyle="--")
        plt.plot(part[TIME_COL], part["cnn1d_pred_voltage"], label="normal model: 1D-CNN", linestyle=":")
        plt.title(f"Degraded Validation Curve vs Normal Model at {c_rate}C")
        plt.xlabel("Discharge time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / f"degraded_curve_residual_{c_rate}C.png", dpi=170)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model_name in zip(axes, ["DNN", "1D-CNN"]):
        part = curve_scores[curve_scores["model"] == model_name]
        sns.boxplot(data=part, x="is_degraded", y="tail_positive_residual_v", ax=ax)
        sns.stripplot(data=part, x="is_degraded", y="tail_positive_residual_v", color="black", size=4, alpha=0.65, ax=ax)
        ax.set_title(f"{model_name}: residual score separates degradation")
        ax.set_xlabel("is degraded")
        ax.set_ylabel("Tail positive residual (V)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "degradation_residual_scores.png", dpi=170)
    plt.close()

    err_df = valid_df[[C_RATE_COL, "is_degraded"]].copy().rename(columns={C_RATE_COL: "c_rate"})
    err_df["DNN_positive_residual"] = valid_df["dnn_positive_residual_v"]
    err_df["1D-CNN_positive_residual"] = valid_df["cnn1d_positive_residual_v"]
    err_long = err_df.melt(id_vars=["c_rate", "is_degraded"], var_name="model", value_name="positive_residual_v")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=err_long, x="c_rate", y="positive_residual_v", hue="is_degraded")
    plt.title("Positive Residual by C-rate: normal model predicts higher voltage for degraded cells")
    plt.ylabel("Positive residual (V)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "positive_residual_by_c_rate.png", dpi=170)
    plt.close()

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "normal_fit_metrics": metrics_df.to_dict(orient="records"),
                "degradation_detection_metrics": detection_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Normal validation fit:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nDegradation detection:")
    print(detection_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved evaluation reports to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
