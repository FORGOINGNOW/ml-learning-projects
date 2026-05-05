# ML Learning Projects

This repository contains compact, runnable machine-learning projects built around synthetic data generation, neural-network modeling, training, evaluation, anomaly detection, and visual reporting.

## Projects

### 1. `vision_end2end_driving`

An end-to-end visual driving mini project:

- synthetic front-camera road image generation
- data cleaning and train/validation/test manifests
- DAVE-2 style CUDA CNN training
- CSV/TensorBoard-compatible training logs
- evaluation plots and saliency explanations
- HTML report under `reports/index.html`

Quick run:

```powershell
cd vision_end2end_driving
python src\run_pipeline.py --samples 3000
```

### 2. `battery_discharge_cureve_fitting`

A lithium battery discharge curve fitting project:

- synthetic discharge data from 0.5C to 6C
- validation curves with degraded battery capacity
- DNN and 1D-CNN voltage fitting
- tail-weighted loss for end-of-discharge accuracy
- curve visualizations and HTML report

Quick run:

```powershell
cd battery_discharge_cureve_fitting
python src\run_pipeline.py
```

### 3. `cdu_anomaly_detection`

An unsupervised industrial telemetry anomaly detection project:

- feature engineering for CDU/CPU/NPU temperature and power signals
- mechanism-aware grouping by `a`, `b`, `c`
- IsolationForest baseline
- CUDA AutoEncoder neural anomaly detector
- groupwise percentile ensemble
- validation using hidden `is_abnormal` labels only after training

Quick run:

```powershell
cd cdu_anomaly_detection
python src\cdu_anomaly_detection.py --data cdu_data.csv
```

The large raw CDU data file is not committed; place it locally before running.

### 4. `battery_nn_multiclass_experiment`

A condition-aware battery state classification project:

- synthetic multi-cell BMS time-series generation
- feature engineering by device, date, and operating condition
- separate MLP classifiers for charge and discharge samples
- sample-level and device-level evaluation
- normal-vs-abnormal visual comparisons
- per-anomaly-type feature and time-series plots
- HTML report under `reports/index.html`

Quick run:

```powershell
cd battery_nn_multiclass_experiment
python src\run_pipeline.py
```

Quick smoke run:

```powershell
cd battery_nn_multiclass_experiment
python tests\smoke_test.py
python src\run_pipeline.py --train-devices 25 --valid-devices 10 --test-devices 10 --days 1 --epochs 2
```

## Environment

Python 3.10+ is recommended. Install dependencies:

```powershell
pip install -r requirements.txt
```

CUDA PyTorch is recommended for the training scripts. If your machine already has CUDA-enabled PyTorch installed, keep that version.

Optional TensorBoard:

```powershell
pip install tensorboard
tensorboard --logdir runs
```

## Notes

The generated datasets and reports included here are intentionally small enough for learning and review. Large checkpoints, cache folders, and transient training logs are ignored by `.gitignore` and can be regenerated with each project's `run_pipeline.py`.
