import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
LABEL_COL = "is_abnormal"
ID_COL = "device_id"
CAT_COLS = ["a", "b", "c"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_cdu(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip()
    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    temp_cols = [f"npu_temp_{i}" for i in range(1, 9)]
    power_cols = [f"npu_power_{i}" for i in range(1, 9)]

    out["npu_temp_mean"] = out[temp_cols].mean(axis=1)
    out["npu_temp_std"] = out[temp_cols].std(axis=1)
    out["npu_temp_max"] = out[temp_cols].max(axis=1)
    out["npu_temp_min"] = out[temp_cols].min(axis=1)
    out["npu_temp_range"] = out["npu_temp_max"] - out["npu_temp_min"]
    out["npu_power_mean"] = out[power_cols].mean(axis=1)
    out["npu_power_std"] = out[power_cols].std(axis=1)
    out["npu_power_max"] = out[power_cols].max(axis=1)
    out["npu_power_min"] = out[power_cols].min(axis=1)
    out["npu_power_range"] = out["npu_power_max"] - out["npu_power_min"]
    out["cdu_temp_lift"] = out["CDU_temp_out"] - out["CDU_temp_in"]
    out["cpu_to_npu_temp_gap"] = out["CPU_max_temp"] - out["npu_temp_mean"]
    out["cpu_power_to_cdu_delta_ratio"] = out["CPU_max_power"] / (out["CDU_power_delta"].abs() + 1e-6)
    out["npu_power_to_cdu_delta_ratio"] = out["npu_power_mean"] / (out["CDU_power_delta"].abs() + 1e-6)

    for i in range(1, 9):
        out[f"npu_temp_{i}_dev"] = out[f"npu_temp_{i}"] - out["npu_temp_mean"]
        out[f"npu_power_{i}_dev"] = out[f"npu_power_{i}"] - out["npu_power_mean"]
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    label = df[LABEL_COL].astype(int) if LABEL_COL in df.columns else None
    feature_df = df.drop(columns=[c for c in [LABEL_COL, ID_COL] if c in df.columns])
    feature_df = add_physics_features(feature_df)
    return feature_df, label


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in feature_df.columns if c not in CAT_COLS]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        latent = max(12, min(64, input_dim // 3))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_rows <= max_rows:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def train_isolation_forest(x_train: np.ndarray, contamination: float) -> IsolationForest:
    model = IsolationForest(
        n_estimators=220,
        max_samples=min(8192, len(x_train)),
        contamination=contamination,
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(x_train)
    return model


def train_autoencoder(
    x_train: np.ndarray,
    input_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[AutoEncoder, list[dict[str, float]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. AutoEncoder training requires CUDA.")

    device = torch.device("cuda")
    model = AutoEncoder(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        history.append({"epoch": epoch, "train_reconstruction_loss": total_loss / len(dataset)})
    return model, history


def autoencoder_scores(model: AutoEncoder, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    model.eval()
    scores = []
    feature_errors = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            reconstructed = model(batch_x)
            err = (reconstructed - batch_x).pow(2)
            scores.append(err.mean(dim=1).detach().cpu().numpy())
            feature_errors.append(err.detach().cpu().numpy())
    feature_errors_arr = np.vstack(feature_errors)
    return np.concatenate(scores), feature_errors_arr


def robust_minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(values, [0.001, 0.999])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def groupwise_percentile_scores(scores: np.ndarray, groups: pd.Series) -> np.ndarray:
    percentiles = np.zeros(len(scores), dtype=np.float32)
    score_series = pd.Series(scores)
    for _, idx in score_series.groupby(groups, sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        ranks = pd.Series(scores[idx_arr]).rank(method="average", pct=True).to_numpy()
        percentiles[idx_arr] = ranks
    return percentiles


def groupwise_top_flags(scores: np.ndarray, groups: pd.Series, contamination: float) -> np.ndarray:
    score_series = pd.Series(scores)
    flags = np.zeros(len(scores), dtype=int)
    for _, idx in score_series.groupby(groups, sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        n_flag = max(1, int(math.ceil(len(idx_arr) * contamination)))
        top_idx = idx_arr[np.argpartition(scores[idx_arr], -n_flag)[-n_flag:]]
        flags[top_idx] = 1
    return flags


def evaluate(y_true: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "predicted_anomalies": int(pred.sum()),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_plots(
    df: pd.DataFrame,
    results: pd.DataFrame,
    metrics_df: pd.DataFrame,
    ae_history: list[dict[str, float]],
    feature_errors: np.ndarray,
    feature_names: np.ndarray,
    out_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=LABEL_COL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Abnormal"])
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Hidden Label Distribution (Only Used for Evaluation)")
    plt.tight_layout()
    plt.savefig(out_dir / "01_hidden_label_distribution.png", dpi=160)
    plt.close()

    group_counts = df.groupby(CAT_COLS).size().reset_index(name="rows")
    for c_value, part in group_counts.groupby("c"):
        pivot = part.pivot(index="a", columns="b", values="rows")
        plt.figure(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="crest")
        plt.title(f"Mechanism Group Size: c={c_value}")
        plt.tight_layout()
        plt.savefig(out_dir / f"02_group_size_{c_value}.png", dpi=160)
        plt.close()

    score_cols = ["iforest_score", "autoencoder_score", "ensemble_score"]
    plot_results = results
    if len(plot_results) > 120_000:
        plot_results = plot_results.sample(n=120_000, random_state=SEED)
    score_long = plot_results.melt(
        id_vars=[LABEL_COL],
        value_vars=score_cols,
        var_name="score_type",
        value_name="score",
    )
    g = sns.FacetGrid(score_long, col="score_type", hue=LABEL_COL, sharex=False, sharey=False, height=3.5)
    g.map_dataframe(sns.kdeplot, x="score", common_norm=False)
    g.add_legend(title=LABEL_COL)
    g.fig.suptitle("Score Distribution by Hidden Label", y=1.05)
    plt.savefig(out_dir / "03_score_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()

    metric_cols = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    metrics_long = metrics_df.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="value")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metrics_long, x="metric", y="value", hue="model")
    plt.ylim(0, 1)
    plt.title("Anomaly Detection Metrics")
    plt.tight_layout()
    plt.savefig(out_dir / "04_metric_comparison.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, model_name in zip(axes, ["IsolationForest", "CUDA AutoEncoder", "Groupwise Ensemble"]):
        pred_col = {
            "IsolationForest": "iforest_pred",
            "CUDA AutoEncoder": "autoencoder_pred",
            "Groupwise Ensemble": "ensemble_pred",
        }[model_name]
        cm = confusion_matrix(results[LABEL_COL], results[pred_col], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticklabels(["Normal", "Abnormal"])
    plt.tight_layout()
    plt.savefig(out_dir / "05_confusion_matrices.png", dpi=160)
    plt.close()

    if ae_history:
        hist = pd.DataFrame(ae_history)
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=hist, x="epoch", y="train_reconstruction_loss", marker="o")
        plt.title("CUDA AutoEncoder Training Loss")
        plt.tight_layout()
        plt.savefig(out_dir / "06_autoencoder_training_loss.png", dpi=160)
        plt.close()

    top_err = pd.DataFrame(
        {"feature": feature_names, "mean_reconstruction_error": feature_errors.mean(axis=0)}
    ).sort_values("mean_reconstruction_error", ascending=False).head(24)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top_err, x="mean_reconstruction_error", y="feature", color="#4C78A8")
    plt.title("AutoEncoder Feature Reconstruction Error")
    plt.tight_layout()
    plt.savefig(out_dir / "07_autoencoder_feature_error.png", dpi=160)
    plt.close()

    top_by_group = (
        results.groupby(CAT_COLS)["ensemble_pred"]
        .sum()
        .reset_index(name="predicted_anomalies")
    )
    for c_value, part in top_by_group.groupby("c"):
        pivot = part.pivot(index="a", columns="b", values="predicted_anomalies")
        plt.figure(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="flare")
        plt.title(f"Predicted Anomalies by Mechanism: c={c_value}")
        plt.tight_layout()
        plt.savefig(out_dir / f"08_predicted_anomalies_{c_value}.png", dpi=160)
        plt.close()


def save_html_report(metrics_df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    images = sorted(p.name for p in out_dir.glob("*.png"))
    image_cards = "\n".join(
        f'<article class="card"><h3>{img[:-4].replace("_", " ")}</h3><img src="{img}" alt="{img}"></article>'
        for img in images
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CDU Anomaly Detection Report</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #f7f8fa; color: #1f2933; }}
    header {{ padding: 28px 32px 18px; background: #fff; border-bottom: 1px solid #d8dee8; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #657282; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px 20px 40px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .stat, .card, .table-wrap {{ background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }}
    .stat {{ padding: 14px 16px; }}
    .stat span {{ display: block; color: #657282; font-size: 13px; }}
    .stat strong {{ display: block; margin-top: 6px; font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; padding: 12px; margin-bottom: 22px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d8dee8; padding: 10px 12px; text-align: left; white-space: nowrap; }}
    th {{ color: #657282; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ padding: 14px; }}
    .card h3 {{ margin: 0 0 10px; font-size: 15px; color: #657282; }}
    .card img {{ width: 100%; height: auto; display: block; border: 1px solid #edf0f4; border-radius: 4px; background: white; }}
    @media (max-width: 560px) {{ header {{ padding: 22px 18px 14px; }} main {{ padding: 18px 12px 32px; }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>CDU 异常检测报告</h1>
    <p>训练阶段不使用 is_abnormal；标签仅用于最终评估。</p>
  </header>
  <main>
    <div class="summary">
      <div class="stat"><span>Rows</span><strong>{summary["rows"]}</strong></div>
      <div class="stat"><span>Mechanism Groups</span><strong>{summary["mechanism_groups"]}</strong></div>
      <div class="stat"><span>Expected Rate</span><strong>{summary["contamination"]:.6f}</strong></div>
      <div class="stat"><span>CUDA Device</span><strong>{summary["cuda_device"]}</strong></div>
    </div>
    <section class="table-wrap">{metrics_df.to_html(index=False, float_format=lambda x: f"{x:.6f}")}</section>
    <section class="grid">{image_cards}</section>
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised CDU anomaly detection with ML and CUDA neural network.")
    parser.add_argument("--data", type=Path, default=Path("cdu_data.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "cdu_anomaly_detection")
    parser.add_argument("--contamination", type=float, default=5e-5)
    parser.add_argument("--iforest-train-rows", type=int, default=300_000)
    parser.add_argument("--ae-train-rows", type=int, default=500_000)
    parser.add_argument("--ae-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_cdu(args.data)
    feature_df, labels = build_feature_matrix(df)
    if labels is None:
        raise ValueError("Expected is_abnormal for evaluation, but it will not be used during training.")
    y_true = labels.to_numpy(dtype=int)
    groups = df[CAT_COLS].astype(str).agg("|".join, axis=1)

    preprocessor = build_preprocessor(feature_df)
    fit_idx = sample_indices(len(feature_df), min(args.iforest_train_rows, args.ae_train_rows), SEED)
    preprocessor.fit(feature_df.iloc[fit_idx])
    x_all = preprocessor.transform(feature_df).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out()

    if_idx = sample_indices(len(x_all), args.iforest_train_rows, SEED)
    iforest = train_isolation_forest(x_all[if_idx], args.contamination)
    iforest_score = -iforest.score_samples(x_all)

    ae_idx = sample_indices(len(x_all), args.ae_train_rows, SEED + 1)
    ae_model, ae_history = train_autoencoder(
        x_all[ae_idx],
        input_dim=x_all.shape[1],
        epochs=args.ae_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    ae_score, feature_errors = autoencoder_scores(ae_model, x_all, args.batch_size)

    iforest_rank = groupwise_percentile_scores(iforest_score, groups)
    ae_rank = groupwise_percentile_scores(ae_score, groups)
    ensemble_score = 0.5 * iforest_rank + 0.5 * ae_rank

    iforest_pred = groupwise_top_flags(iforest_score, groups, args.contamination)
    ae_pred = groupwise_top_flags(ae_score, groups, args.contamination)
    ensemble_pred = groupwise_top_flags(ensemble_score, groups, args.contamination)

    results = df[[ID_COL, *CAT_COLS, LABEL_COL]].copy()
    results["iforest_score"] = iforest_score
    results["autoencoder_score"] = ae_score
    results["ensemble_score"] = ensemble_score
    results["iforest_pred"] = iforest_pred
    results["autoencoder_pred"] = ae_pred
    results["ensemble_pred"] = ensemble_pred

    metrics_df = pd.DataFrame(
        [
            {"model": "IsolationForest", **evaluate(y_true, iforest_score, iforest_pred)},
            {"model": "CUDA AutoEncoder", **evaluate(y_true, ae_score, ae_pred)},
            {"model": "Groupwise Ensemble", **evaluate(y_true, ensemble_score, ensemble_pred)},
        ]
    ).sort_values("f1", ascending=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "mechanism_groups": int(groups.nunique()),
        "hidden_label_anomalies": int(y_true.sum()),
        "hidden_label_rate": float(y_true.mean()),
        "contamination": float(args.contamination),
        "encoded_feature_count": int(x_all.shape[1]),
        "cuda_device": torch.cuda.get_device_name(0),
        "best_model_by_f1": metrics_df.iloc[0]["model"],
        "best_f1": float(metrics_df.iloc[0]["f1"]),
    }

    metrics_df.to_csv(args.out_dir / "metrics.csv", index=False)
    results.sort_values("ensemble_score", ascending=False).head(500).to_csv(args.out_dir / "top_anomalies.csv", index=False)
    pd.DataFrame(ae_history).to_csv(args.out_dir / "autoencoder_training_history.csv", index=False)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_plots(df, results, metrics_df, ae_history, feature_errors, feature_names, args.out_dir)
    save_html_report(metrics_df, summary, args.out_dir)

    print("Saved outputs to:", args.out_dir.resolve())
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("CUDA device:", summary["cuda_device"])


if __name__ == "__main__":
    main()
