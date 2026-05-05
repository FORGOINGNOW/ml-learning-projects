"""模型评估脚本：在指定 split 上生成预测结果、指标、分类报告和混淆矩阵。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from battery_classifier.inference import aggregate_device_predictions, load_model_index, predict_feature_table
from battery_classifier.utils import ensure_dirs, project_root


def _metric_row(scope: str, predictions: pd.DataFrame) -> dict:
    return {
        "scope": scope,
        "rows": int(len(predictions)),
        "accuracy": float(accuracy_score(predictions["state_label"], predictions["y_pred"])),
        "macro_f1": float(f1_score(predictions["state_label"], predictions["y_pred"], average="macro", zero_division=0)),
    }


def _plot_confusion(predictions: pd.DataFrame, labels: list[str], title: str, output: Path) -> None:
    matrix = confusion_matrix(predictions["state_label"], predictions["y_pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained battery state classifiers.")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    root = project_root()
    ensure_dirs(root)
    reports_dir = root / "reports" / "model"
    reports_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(root / args.features)
    index = load_model_index(root)
    class_names = list(index["class_names"])

    sample_preds = predict_feature_table(features, root, split=args.split)
    device_preds = aggregate_device_predictions(sample_preds, class_names)
    sample_preds.to_csv(reports_dir / "sample_predictions.csv", index=False)
    device_preds.to_csv(reports_dir / "device_predictions.csv", index=False)

    metric_rows = [_metric_row(f"{args.split}_samples_all_conditions", sample_preds)]
    for condition, group in sample_preds.groupby("condition"):
        metric_rows.append(_metric_row(f"{args.split}_samples_{condition}", group))
    metric_rows.append(_metric_row(f"{args.split}_devices_ensemble", device_preds))
    pd.DataFrame(metric_rows).to_csv(reports_dir / "metrics.csv", index=False)

    reports = {
        "sample_level": classification_report(
            sample_preds["state_label"],
            sample_preds["y_pred"],
            labels=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "device_level": classification_report(
            device_preds["state_label"],
            device_preds["y_pred"],
            labels=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    (reports_dir / "classification_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_confusion(sample_preds, class_names, "Sample-level confusion matrix", reports_dir / "confusion_matrix_sample.png")
    _plot_confusion(device_preds, class_names, "Device-level confusion matrix", reports_dir / "confusion_matrix_device.png")
    print(f"Saved evaluation outputs: {reports_dir}")


if __name__ == "__main__":
    main()
