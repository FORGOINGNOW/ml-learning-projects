# CDU Anomaly Detection

This project demonstrates unsupervised anomaly detection for CDU telemetry.

The original local data file is large and is not committed:

- `cdu_data.csv`
- `cdu_data.parquet`
- `cdu_data_iforest_result.parquet`

Place `cdu_data.csv` in this project folder, or pass a custom path with `--data`.

## Data Assumption

The dataset contains:

- `device_id`
- mechanism/category columns: `a`, `b`, `c`
- CPU/CDU/NPU temperature and power features
- `is_abnormal`, which is treated as a hidden evaluation label only

Training does not use `is_abnormal`. The label is only used after scoring to evaluate whether the unsupervised detector found the injected anomalies.

## Methods

- `IsolationForest`
- CUDA AutoEncoder
- groupwise ensemble using per-mechanism percentile ranking

Because `a`, `b`, and `c` represent different physical mechanisms, anomaly thresholds are applied groupwise by `(a, b, c)` instead of one global cutoff.

## Run

```powershell
python src\cdu_anomaly_detection.py --data cdu_data.csv
```

Outputs are written to:

```text
reports/
```

The included `reports/` folder contains the latest local run summary and visualizations.
