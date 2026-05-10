import argparse
import html
from pathlib import Path

import pandas as pd


def card(title: str, rel: str) -> str:
    return f'<article class="card"><h3>{html.escape(title)}</h3><img src="{html.escape(rel)}" alt="{html.escape(title)}"></article>'


def table_or_note(path: Path, note: str) -> str:
    if path.exists():
        return pd.read_csv(path).to_html(index=False, float_format=lambda x: f"{x:.5f}")
    return f"<p>{html.escape(note)}</p>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HTML report for battery discharge fitting project.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reports/index.html"))
    args = parser.parse_args()
    root = args.root
    reports = root / "reports"

    fit_metrics_html = table_or_note(
        reports / "model" / "valid_metrics_normal_only.csv",
        "No normal validation metrics yet.",
    )
    detection_metrics_html = table_or_note(
        reports / "model" / "degradation_detection_metrics.csv",
        "No degradation detection metrics yet.",
    )

    imgs = []
    for title, rel in [
        ("Normal Training Curves by C-rate", "data/discharge_curves_by_c_rate.png"),
        ("Feature Overview", "data/data_feature_overview.png"),
        ("Normal Train vs Mixed Validation", "data/train_valid_curve_comparison.png"),
        ("Degradation Residual Scores", "model/degradation_residual_scores.png"),
        ("Positive Residual by C-rate", "model/positive_residual_by_c_rate.png"),
    ]:
        if (reports / rel).exists():
            imgs.append(card(title, rel))
    for img in sorted((reports / "model").glob("degraded_curve_residual_*C.png")):
        imgs.append(card(img.stem.replace("_", " "), f"model/{img.name}"))

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Battery Discharge Degradation Detection</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #f7f8fa; color: #202832; }}
    header {{ padding: 28px 32px 18px; background: #fff; border-bottom: 1px solid #d8dee8; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #657282; line-height: 1.55; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px 20px 40px; }}
    h2 {{ margin: 26px 0 12px; font-size: 20px; }}
    .table-wrap, .card, .note {{ background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }}
    .table-wrap, .note {{ padding: 12px; overflow-x: auto; }}
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
    <h1>锂电池放电衰减检测实验</h1>
    <p>训练集只包含正常电池曲线；验证集混入实际容量衰减曲线。模型学习正常放电电压，衰减电池通过“正常模型预测电压高于实际电压”的尾部正残差被识别。</p>
  </header>
  <main>
    <section class="note">
      <p><strong>Core Logic:</strong> DNN and 1D-CNN are trained only on normal discharge curves. Validation degradation is not a fitting target; it is detected as a residual pattern against the normal model.</p>
    </section>
    <h2>Normal Validation Fit</h2>
    <section class="table-wrap">{fit_metrics_html}</section>
    <h2>Degradation Detection</h2>
    <section class="table-wrap">{detection_metrics_html}</section>
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
