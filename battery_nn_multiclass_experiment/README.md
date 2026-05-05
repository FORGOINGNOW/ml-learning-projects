# Battery NN Multiclass Experiment

This project is a compact battery-state classification experiment built around synthetic BMS time-series data and condition-aware neural networks.

The current baseline focuses on a complete, runnable workflow:

- generate constrained synthetic BMS readings for multi-cell battery devices
- build fixed-length features by device, date, and operating condition
- train separate MLP classifiers for charge and discharge samples
- evaluate both sample-level and device-level predictions
- generate data-quality plots, anomaly comparison plots, confusion matrices, metrics, and an HTML report

## Data Assumptions

Raw data uses the following schema:

```text
time
devices
battery_cell_temp_1 ... battery_cell_temp_8
battery_cell_volt_1 ... battery_cell_volt_8
battery_cell_SOC_1 ... battery_cell_SOC_8
current
split
state_label
```

`split` and `state_label` are experiment labels. Real inference data can omit them; `src/predict.py` will treat missing labels as `inference` and `unknown`.

Default settings are in `configs/default.json`:

- 200 training devices, 50 validation devices, and 50 test devices
- labels: `normal`, `outliers`, `level_shift`, `gradual_drift`, and `battery_replacement`
- one sample every two minutes
- two days of readings per device
- eight cells per device
- cell temperature constrained to `[20, 60]` degrees Celsius
- cell voltage constrained to `3.5V`
- rated capacity set to `17Ah`
- discharge current centered around `-0.5A`; charge current is positive

Operating conditions are inferred from current:

```text
current > +0.05A  => charge
current < -0.05A  => discharge
otherwise         => idle
```

The training pipeline filters out `idle` rows and trains separate classifiers for `charge` and `discharge`. During prediction, the active condition selects the matching model, and device-level results are ensembled from sample probabilities.

## Quick Start

From this project directory:

```powershell
python src\run_pipeline.py
```

Quick smoke run:

```powershell
python tests\smoke_test.py
python src\run_pipeline.py --train-devices 25 --valid-devices 10 --test-devices 10 --days 1 --epochs 2
```

Main outputs:

```text
data/raw/simulated_battery_readings.csv
data/raw/device_metadata.csv
data/processed/features.csv
runs/models/
reports/model/metrics.csv
reports/model/sample_predictions.csv
reports/model/device_predictions.csv
reports/index.html
```

Generated CSV files, model artifacts, reports, and runtime caches are ignored by default because they can be regenerated.

## Visualizations

`src/visualize_data.py` creates exploratory plots under `reports/data/`, including:

- label and operating-condition distributions
- temperature and voltage distributions
- normal-vs-abnormal feature boxplots
- standardized abnormal-minus-normal feature gaps
- raw time-series examples for normal and abnormal devices
- per-abnormal-type deviation heatmaps
- per-state boxplots and time-series comparisons across all anomaly types

These plots are linked from `reports/index.html` by `src/make_report.py`.

## Predicting Real Data

After training a model with synthetic or real labeled data, run:

```powershell
python src\predict.py --raw path\to\real_battery_data.csv --output-dir reports\predictions
```

Real data should include at least:

```text
time, devices, battery_cell_temp_1..8, battery_cell_volt_1..8, battery_cell_SOC_1..8, current
```

## Project Structure

```text
battery_nn_multiclass_experiment/
  configs/
  data/
  reports/
  runs/
  src/
    battery_classifier/
    simulate_data.py
    build_features.py
    visualize_data.py
    train.py
    evaluate.py
    predict.py
    make_report.py
    run_pipeline.py
  tests/
  README.md
  requirements.txt
```

## Future Questions

To move this baseline closer to a real production workflow, clarify:

- whether real labels are device-level, cell-level, day-level, or time-window-level
- whether `battery_replacement` should be represented by SOC jumps, voltage plateau shifts, capacity recovery, maintenance logs, or a combination
- the expected charge-current range and charging policy
- whether idle or standby periods should become a separate model condition
- whether train, validation, and test splits must always be grouped by device
- whether prediction output should be one label per device, day, condition, cell, or time window
