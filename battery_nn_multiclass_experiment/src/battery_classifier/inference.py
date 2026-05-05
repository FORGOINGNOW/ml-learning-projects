"""推理模块：加载已训练的分工况模型，输出样本级和设备级分类结果。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from battery_classifier.features import feature_columns
from battery_classifier.model import MLPClassifier


def load_model_index(root: Path) -> dict:
    index_path = root / "runs" / "models" / "model_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Model index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _load_condition_model(root: Path, entry: dict, device: torch.device) -> tuple[MLPClassifier, object]:
    checkpoint = torch.load(root / entry["model_path"], map_location=device, weights_only=False)
    model = MLPClassifier(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    scaler = joblib.load(root / entry["scaler_path"])
    return model, scaler


def predict_feature_table(features: pd.DataFrame, root: Path, split: str | None = None) -> pd.DataFrame:
    index = load_model_index(root)
    class_names = list(index["class_names"])
    columns = list(index.get("feature_columns", feature_columns(features)))
    if split is not None:
        features = features[features["split"] == split].copy()
    if features.empty:
        raise ValueError("No feature rows available for prediction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames: list[pd.DataFrame] = []
    for condition, entry in index["conditions"].items():
        subset = features[features["condition"] == condition].copy()
        if subset.empty:
            continue
        model, scaler = _load_condition_model(root, entry, device)
        x = scaler.transform(subset[columns].to_numpy(dtype=np.float32))
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32, device=device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_idx = probs.argmax(axis=1)
        out = subset[["split", "devices", "state_label", "date", "condition"]].copy()
        out["y_pred"] = [class_names[idx] for idx in pred_idx]
        for idx, class_name in enumerate(class_names):
            out[f"proba_{class_name}"] = probs[:, idx]
        frames.append(out)

    if not frames:
        raise ValueError("No condition-specific model matched the feature table.")
    return pd.concat(frames, ignore_index=True)


def aggregate_device_predictions(sample_predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    proba_cols = [f"proba_{name}" for name in class_names]
    group_cols = ["split", "devices", "state_label"]
    agg = sample_predictions.groupby(group_cols, as_index=False)[proba_cols].mean()
    pred_idx = agg[proba_cols].to_numpy().argmax(axis=1)
    agg["y_pred"] = [class_names[idx] for idx in pred_idx]
    agg["sample_rows"] = sample_predictions.groupby(group_cols).size().to_numpy()
    return agg
