# Battery Discharge Curve Fitting

This project uses neural networks to learn **normal lithium battery discharge behavior** and then identify degraded validation cells from residual patterns.

The directory name keeps the original spelling:

```text
battery_discharge_cureve_fitting
```

## Core Logic

The training set contains only normal discharge curves. It does **not** contain degraded or abnormal battery curves.

The validation set contains a mixture of:

- normal validation curves
- degraded curves whose actual available capacity is lower, about 82% of the rated capacity

The model is trained as a normal-behavior model. When a degraded cell appears in validation, its measured voltage drops earlier than the normal model expects. The degradation signal is therefore the positive residual:

```text
normal_model_predicted_voltage - actual_voltage
```

A large positive residual near the tail of discharge means the cell is declining faster than the normal model predicts.

## Data

Both `discharge_df.csv` and `valid_df.csv` keep the requested four columns:

```text
放电时间, 放电倍率, 放电电压, 已放电容量
```

Default rated capacity:

```text
17Ah
```

C-rate coverage:

```text
0.5C, 1C, 1.5C, 2C, 3C, 4C, 5C, 6C
```

Additional curve-level metadata is written separately:

- `data/train_curve_meta.csv`
- `data/valid_curve_meta.csv`

These metadata files mark which validation curves are degraded. They are used only for evaluation and plotting, not as model inputs.

## Models

- `DNN`: point-wise voltage regression model.
- `1D-CNN`: sequence model that predicts the full voltage curve.

The target is:

```text
y = 放电电压
```

The model inputs are derived only from rated-capacity features:

- `rated_time_frac`
- `c_rate`
- `capacity_norm`
- `soc_est`
- `current_a`
- `c_rate_sq`
- `sqrt_capacity_norm`
- `tail_progress`

No actual curve max capacity, SOH proxy, or degraded label is passed into the model.

## Run

```powershell
python src\run_pipeline.py
```

Quick verification:

```powershell
python src\run_pipeline.py --epochs 5
```

## Outputs

- `data/discharge_df.csv`
- `data/valid_df.csv`
- `reports/data/discharge_curves_by_c_rate.png`
- `reports/model/valid_metrics_normal_only.csv`
- `reports/model/degradation_detection_metrics.csv`
- `reports/model/curve_degradation_scores.csv`
- `reports/model/degraded_curve_residual_*C.png`
- `reports/model/degradation_residual_scores.png`
- `reports/model/positive_residual_by_c_rate.png`
- `reports/index.html`
- `runs/dnn_best.pt`
- `runs/cnn1d_best.pt`

If TensorBoard is installed:

```powershell
pip install tensorboard
tensorboard --logdir runs\fit_logs
```
