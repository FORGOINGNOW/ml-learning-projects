import argparse
import html
from pathlib import Path

import pandas as pd


def card(title: str, rel: str) -> str:
    return f'<article class="card"><h3>{html.escape(title)}</h3><img src="{html.escape(rel)}" alt="{html.escape(title)}"></article>'


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HTML report for battery discharge fitting project.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reports/index.html"))
    args = parser.parse_args()
    root = args.root
    reports = root / "reports"
    metrics_path = reports / "model" / "valid_metrics.csv"
    metrics_html = pd.read_csv(metrics_path).to_html(index=False, float_format=lambda x: f"{x:.5f}") if metrics_path.exists() else "<p>No metrics yet.</p>"
    imgs = []
    for title, rel in [
        ("Discharge Curves by C-rate", "data/discharge_curves_by_c_rate.png"),
        ("Feature Overview", "data/data_feature_overview.png"),
        ("Train vs Validation Curves", "data/train_valid_curve_comparison.png"),
        ("Prediction Scatter", "model/prediction_scatter.png"),
        ("Absolute Error by C-rate", "model/abs_error_by_c_rate.png"),
    ]:
        if (reports / rel).exists():
            imgs.append(card(title, rel))
    for img in sorted((reports / "model").glob("curve_fit_*C.png")):
        imgs.append(card(img.stem.replace("_", " "), f"model/{img.name}"))

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Battery Discharge Curve Fitting</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #f7f8fa; color: #202832; }}
    header {{ padding: 28px 32px 18px; background: #fff; border-bottom: 1px solid #d8dee8; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #657282; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px 20px 40px; }}
    h2 {{ margin: 26px 0 12px; font-size: 20px; }}
    .table-wrap, .card {{ background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }}
    .table-wrap {{ padding: 12px; overflow-x: auto; }}
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
    <h1>锂电池放电曲线拟合实验</h1>
    <p>DNN 与 1D-CNN 对不同倍率放电电压曲线的拟合、验证与可视化。</p>
  </header>
  <main>
    <h2>Validation Metrics</h2>
    <section class="table-wrap">{metrics_html}</section>
    <h2>Visual Results</h2>
    <section class="grid">{''.join(imgs)}</section>
  </main>
</body>
</html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(f"Saved report to {(root / args.out).resolve()}")


if __name__ == "__main__":
    main()
