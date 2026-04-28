import argparse
import html
from pathlib import Path

import pandas as pd


def image_card(title: str, rel_path: str) -> str:
    return f"""
    <article class="card">
      <h3>{html.escape(title)}</h3>
      <img src="{html.escape(rel_path)}" alt="{html.escape(title)}">
    </article>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact HTML report for the driving project.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reports/index.html"))
    args = parser.parse_args()

    root = args.root
    reports = root / "reports"
    metrics_path = reports / "model" / "test_metrics.csv"
    metrics_html = "<p>No model metrics yet.</p>"
    if metrics_path.exists():
        metrics_html = pd.read_csv(metrics_path).to_html(index=False, float_format=lambda x: f"{x:.4f}")

    cards = []
    for title, path in [
        ("Label Distributions", "data/label_distributions.png"),
        ("Steering vs Curvature", "data/steering_vs_curvature.png"),
        ("Scenario Rates", "data/scenario_rates.png"),
        ("Sample Grid", "data/sample_grid.png"),
        ("Prediction Scatter", "model/prediction_scatter.png"),
        ("Absolute Error", "model/absolute_error_boxplot.png"),
        ("Worst Steering Examples", "model/worst_steering_examples.png"),
    ]:
        if (reports / path).exists():
            cards.append(image_card(title, path))

    saliency_cards = []
    for img in sorted((reports / "explain").glob("saliency_*.png"))[:12]:
        saliency_cards.append(image_card(img.stem, f"explain/{img.name}"))

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vision End-to-End Driving Report</title>
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
    <h1>视觉端到端自动驾驶小项目</h1>
    <p>从合成数据、清洗、训练、评估到 saliency 解释的完整实验台。</p>
  </header>
  <main>
    <h2>Test Metrics</h2>
    <section class="table-wrap">{metrics_html}</section>
    <h2>Data And Model Results</h2>
    <section class="grid">{''.join(cards)}</section>
    <h2>Model Explanation</h2>
    <section class="grid">{''.join(saliency_cards) if saliency_cards else '<p>No saliency images yet.</p>'}</section>
  </main>
</body>
</html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(f"Saved report to {(root / args.out).resolve()}")


if __name__ == "__main__":
    main()
