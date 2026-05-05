"""模型训练脚本：分别为充电和放电工况训练 MLP 多分类器并保存模型。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from battery_classifier.features import feature_columns
from battery_classifier.model import MLPClassifier
from battery_classifier.utils import ensure_dirs, load_config, project_root, set_seed


def _encode(labels: pd.Series, class_names: list[str]) -> np.ndarray:
    lookup = {name: idx for idx, name in enumerate(class_names)}
    missing = sorted(set(labels) - set(lookup))
    if missing:
        raise ValueError(f"Labels not present in config abnormal_states: {missing}")
    return labels.map(lookup).to_numpy(dtype=np.int64)


def _predict(model: MLPClassifier, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def _evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    probs = _predict(model, x, device)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (num_classes * counts[present])
    weights[~present] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_condition_model(
    condition: str,
    features: pd.DataFrame,
    config: dict,
    root: Path,
    columns: list[str],
    class_names: list[str],
) -> tuple[dict, list[dict]]:
    train_df = features[(features["split"] == "train") & (features["condition"] == condition)].copy()
    valid_df = features[(features["split"] == "valid") & (features["condition"] == condition)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"Condition {condition!r} needs non-empty train and valid feature rows.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[columns].to_numpy(dtype=np.float32))
    x_valid = scaler.transform(valid_df[columns].to_numpy(dtype=np.float32))
    y_train = _encode(train_df["state_label"], class_names)
    y_valid = _encode(valid_df["state_label"], class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        input_dim=len(columns),
        num_classes=len(class_names),
        hidden_dims=list(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, len(class_names), device))
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)

    best_score = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_metrics = _evaluate(model, x_train, y_train, device)
        valid_metrics = _evaluate(model, x_valid, y_valid, device)
        history.append(
            {
                "condition": condition,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
            }
        )
        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["early_stop_patience"]):
            break

    model_path = root / "runs" / "models" / f"{condition}_mlp.pt"
    scaler_path = root / "runs" / "models" / f"{condition}_scaler.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": len(columns),
            "num_classes": len(class_names),
            "hidden_dims": list(config["hidden_dims"]),
            "dropout": float(config["dropout"]),
            "condition": condition,
            "class_names": class_names,
            "feature_columns": columns,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    return (
        {
            "condition": condition,
            "best_epoch": best_epoch,
            "best_valid_macro_f1": float(best_score),
            "model_path": str(model_path.relative_to(root)).replace("\\", "/"),
            "scaler_path": str(scaler_path.relative_to(root)).replace("\\", "/"),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        },
        history,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train condition-aware battery state classifiers.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--features", default="data/processed/features.csv")
    args = parser.parse_args()

    root = project_root()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    ensure_dirs(root)

    features = pd.read_csv(root / args.features)
    columns = feature_columns(features)
    class_names = list(config["abnormal_states"])
    summaries = []
    all_history = []
    model_index = {
        "class_names": class_names,
        "feature_columns": columns,
        "conditions": {},
    }

    for condition in config["conditions"]:
        summary, history = train_condition_model(condition, features, config, root, columns, class_names)
        summaries.append(summary)
        all_history.extend(history)
        model_index["conditions"][condition] = {
            "model_path": summary["model_path"],
            "scaler_path": summary["scaler_path"],
        }
        print(
            f"{condition}: best valid macro-F1={summary['best_valid_macro_f1']:.3f} "
            f"at epoch {summary['best_epoch']}"
        )

    reports_dir = root / "reports" / "model"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_history).to_csv(reports_dir / "training_history.csv", index=False)
    pd.DataFrame(summaries).to_csv(reports_dir / "training_summary.csv", index=False)
    (root / "runs" / "models" / "model_index.json").write_text(
        json.dumps(model_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model index: {root / 'runs' / 'models' / 'model_index.json'}")


if __name__ == "__main__":
    main()
