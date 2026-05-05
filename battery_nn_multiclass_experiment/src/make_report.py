"""报告生成脚本：把训练评估指标和图表汇总成一个轻量 HTML 报告。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_classifier.utils import ensure_dirs, project_root


def _table_html(path: Path) -> str:
    if not path.exists():
        return "<p>Not generated yet.</p>"
    return pd.read_csv(path).to_html(index=False, classes="data-table", float_format=lambda x: f"{x:.4f}")


def _image(path: str, alt: str) -> str:
    return f'<figure><img src="{path}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight HTML experiment report.")
    parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    report_path = root / "reports" / "index.html"

    metrics = _table_html(root / "reports" / "model" / "metrics.csv")
    training = _table_html(root / "reports" / "model" / "training_summary.csv")
    images = [
        _image("data/01_device_label_distribution.png", "device label distribution"),
        _image("data/02_condition_distribution.png", "raw row condition distribution"),
        _image("data/03_feature_condition_distribution.png", "feature row condition distribution"),
        _image("data/04_temperature_distribution.png", "temperature distribution"),
        _image("data/05_voltage_distribution.png", "voltage distribution"),
        _image("data/06_normal_abnormal_feature_boxplots.png", "normal vs abnormal feature boxplots"),
        _image("data/07_normal_abnormal_standardized_gap.png", "normal vs abnormal standardized feature gap"),
        _image("data/08_normal_abnormal_timeseries_example.png", "normal vs abnormal raw time-series example"),
        _image("data/09_abnormal_type_deviation_heatmap.png", "abnormal type deviation from normal"),
        _image("data/10_all_state_feature_boxplots.png", "normal and each abnormal type feature boxplots"),
        _image("data/11_all_state_timeseries_examples.png", "normal and each abnormal type raw time-series examples"),
        _image("model/confusion_matrix_sample.png", "sample-level confusion matrix"),
        _image("model/confusion_matrix_device.png", "device-level confusion matrix"),
    ]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Battery NN Multiclass Experiment</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
    figure {{ margin: 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    figcaption {{ color: #555; font-size: 0.9rem; margin-top: 0.35rem; }}
    .data-table {{ border-collapse: collapse; margin: 12px 0 28px; min-width: 680px; }}
    .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table th {{ background: #f4f6f8; }}
  </style>
</head>
<body>
  <h1>Battery NN Multiclass Experiment</h1>
  <p>Condition-aware neural-network baseline for charge/discharge battery state classification.</p>
  <h2>Evaluation Metrics</h2>
  {metrics}
  <h2>Training Summary</h2>
  {training}
  <h2>Plots</h2>
  <div class="grid">
    {''.join(images)}
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
