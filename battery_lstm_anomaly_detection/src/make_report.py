from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def dataframe_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return format(value, floatfmt)
        return str(value)

    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def image_md(path: str, title: str) -> str:
    return f"### {title}\n\n![{title}]({path})\n"


def make_markdown(root: Path, config: dict) -> str:
    report_dir = root / "reports"
    metrics = pd.read_csv(report_dir / "model" / "metrics_by_split.csv")
    state_metrics = pd.read_csv(report_dir / "model" / "metrics_by_state.csv")
    device_scores = pd.read_csv(report_dir / "model" / "device_scores.csv")
    threshold = json.loads((root / "runs" / "threshold.json").read_text(encoding="utf-8"))

    abnormal_devices = device_scores[device_scores["true_anomaly"]]
    detected_devices = int(abnormal_devices["detected_any"].sum()) if len(abnormal_devices) else 0
    eval_metrics = metrics[metrics["split"].isin(["valid", "test"])]
    normal_window_alarm_rate = float(eval_metrics["fp"].sum() / max((eval_metrics["tn"] + eval_metrics["fp"]).sum(), 1))

    lines = [
        "# Battery LSTM Residual Anomaly Detection Report",
        "",
        "## Objective",
        "",
        "This mini-project trains an LSTM forecaster on normal lithium battery BMS sequences.",
        "At inference time, the model predicts future voltage, temperature, and SOC indicators.",
        "A residual score above the train-normal high-quantile threshold is flagged as anomalous.",
        "",
        "## Pipeline",
        "",
        "1. Simulate physically constrained 8-cell BMS time series.",
        "2. Engineer pack-level features such as cell voltage spread, max temperature, SOC spread, current state, time-of-day phase, and throughput.",
        "3. Train an LSTM next-horizon forecaster using only normal training devices.",
        "4. Score validation and test windows from normalized residuals.",
        "5. Use the train residual quantile as the anomaly threshold and evaluate point-level and device-level detection.",
        "",
        "## Key Configuration",
        "",
        f"- Rated capacity: `{config['rated_capacity_ah']} Ah`",
        f"- Cells: `{config['n_cells']}`",
        f"- Sample interval: `{config['sample_interval_minutes']} minutes`",
        f"- Sequence length: `{config['seq_len']}`",
        f"- Forecast horizon: `{config['forecast_horizon']}`",
        f"- Threshold quantile: `{config['threshold_quantile']}`",
        f"- Learned threshold: `{threshold['threshold']:.4f}`",
        f"- Best epoch: `{threshold['best_epoch']}`",
        "",
        "## Split Metrics",
        "",
        dataframe_markdown(metrics),
        "",
        "## State Metrics",
        "",
        dataframe_markdown(state_metrics),
        "",
        "## Device-Level Summary",
        "",
        f"- Abnormal devices detected: `{detected_devices}/{len(abnormal_devices)}`",
        f"- Normal-window false alarm rate: `{normal_window_alarm_rate:.2%}`",
        "- Device maintenance decisions should add persistence or hysteresis on top of raw window alarms.",
        "",
        image_md("data/device_state_distribution.png", "Device State Distribution"),
        image_md("data/feature_distributions.png", "Feature Distributions"),
        image_md("data/timeseries_examples.png", "Raw Time-Series Examples"),
        image_md("model/training_curve.png", "Training Curve"),
        image_md("model/score_distribution.png", "Residual Score Distribution"),
        image_md("model/test_confusion_matrix.png", "Test Confusion Matrix"),
        image_md("model/residual_contributors.png", "Residual Contributors"),
        image_md("model/anomaly_timeline_examples.png", "Anomaly Timeline Examples"),
        "",
        "## Interpretation",
        "",
        "The detector is intentionally unsupervised with respect to fault labels: labels are only used for validation.",
        "Voltage-minimum, voltage-spread, temperature-maximum, temperature-spread, SOC-mean, and SOC-spread residuals are weighted because they map to common BMS risks: cell sag, imbalance, heat rise, and sensor drift.",
        "The threshold is derived from train-normal residuals, so the operating point is easy to tighten or relax by changing `threshold_quantile`.",
    ]
    return "\n".join(lines)


def make_html(markdown_text: str) -> str:
    try:
        import markdown

        body = markdown.markdown(markdown_text, extensions=["tables"])
    except Exception:
        body = "<pre>" + markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Battery LSTM Residual Anomaly Detection</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px auto; max-width: 1120px; line-height: 1.55; color: #20242a; }}
    h1, h2, h3 {{ color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #d6dbe1; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f6f9; }}
    img {{ max-width: 100%; border: 1px solid #d8dde3; border-radius: 6px; }}
    code {{ background: #f3f6f9; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown and HTML reports.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    markdown_text = make_markdown(root, config)
    report_md = root / "reports" / "report.md"
    report_html = root / "reports" / "index.html"
    report_md.write_text(markdown_text, encoding="utf-8")
    report_html.write_text(make_html(markdown_text), encoding="utf-8")
    print(f"Saved Markdown report: {report_md}")
    print(f"Saved HTML report: {report_html}")


if __name__ == "__main__":
    main()
