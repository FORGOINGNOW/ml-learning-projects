# Battery LSTM Residual Anomaly Detection

This project builds a complete time-series anomaly detection pipeline for lithium battery BMS indicators.

The detector is trained only on normal devices. It learns to forecast future key indicators from recent BMS history, then flags windows where the observed indicators deviate from the forecast by more than a threshold learned from normal training residuals.

## BMS Signals

The synthetic data follows the same broad convention as the existing battery examples in this workspace:

```text
time
device_id
battery_cell_temp_1 ... battery_cell_temp_8
battery_cell_volt_1 ... battery_cell_volt_8
battery_cell_SOC_1 ... battery_cell_SOC_8
current_a
split
state_label
```

Fault labels are generated only for evaluation. Training uses normal devices only.

## Forecast Targets

The LSTM predicts:

```text
cell_voltage_min
cell_voltage_delta
cell_temp_max
cell_temp_delta
cell_soc_mean
cell_soc_delta
```

The anomaly score is a weighted normalized residual across these metrics. The default threshold is the 98.5th percentile of train-normal scores.

## Run

From this project directory:

```powershell
python src\run_pipeline.py
```

Quick smoke run:

```powershell
python src\run_pipeline.py --epochs 4 --train-devices 12 --valid-devices 6 --test-devices 6 --days 1
```

CUDA is required by default. The training script prints the CUDA device name before fitting.

## Outputs

```text
data/raw/simulated_bms_readings.csv
data/raw/device_metadata.csv
data/processed/features.csv
runs/models/lstm_forecaster_best.pt
runs/threshold.json
runs/train_history.csv
reports/model/window_predictions.csv
reports/model/metrics_by_split.csv
reports/model/metrics_by_state.csv
reports/model/device_scores.csv
reports/report.md
reports/index.html
```
