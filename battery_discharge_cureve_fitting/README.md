# Battery Discharge Curve Fitting

这是一个基于神经网络的锂电池放电曲线拟合小项目，目录名沿用用户要求：

`battery_discharge_cureve_fitting`

项目目标：

- 生成合成放电数据 `discharge_df.csv`
- 生成包含容量衰减电池的验证集 `valid_df.csv`
- 使用 DNN 和 1D-CNN 拟合放电电压
- 可视化不同倍率下的放电曲线
- 输出验证指标、预测曲线、误差分析和 HTML 报告

## 数据字段

训练集和验证集都保持相同列名：

```text
放电时间, 放电倍率, 放电电压, 已放电容量
```

默认额定容量为 `17Ah`，放电倍率覆盖：

```text
0.5C, 1C, 1.5C, 2C, 3C, 4C, 5C, 6C
```

验证集 `valid_df.csv` 中每个倍率至少包含一条容量衰减曲线，容量约为额定容量的 82%，用于观察模型对退化电池的外推能力。

训练集也包含一部分轻重不等的容量衰减曲线。这样做是因为如果训练集中完全没有衰减样本，模型只按额定容量估计 SOC，会在验证集尾部系统性预测偏高。

## 一键运行

在本目录执行：

```powershell
python src\run_pipeline.py
```

快速验证：

```powershell
python src\run_pipeline.py --epochs 5
```

## 模型

- `DNN`：逐点回归模型，每个采样点输入特征，输出该点电压。
- `1D-CNN`：整条曲线序列模型，输入一条放电曲线的特征序列，输出整条电压序列。

原始表中 `y = 放电电压`，其余为基础输入。代码内部会构造派生特征：

- `time_norm`
- `c_rate`
- `capacity_norm`
- `soc_est`
- `current_a`
- `c_rate_sq`
- `sqrt_time_norm`

后续版本额外加入了曲线级特征，专门改善尾部拟合：

- `curve_time_frac`：单条曲线内部的时间进度
- `curve_capacity_frac`：单条曲线内部的容量进度
- `curve_soc_est`：基于单条曲线容量的 SOC 估计
- `soh_proxy`：本条曲线最大放电容量 / 额定容量
- `tail_progress`：尾部区间权重特征

训练时对曲线尾部使用更高 loss 权重，因为放电平台区点数多，普通 MSE 很容易把尾部陡降平均掉。

## 输出

- `data/discharge_df.csv`
- `data/valid_df.csv`
- `reports/data/discharge_curves_by_c_rate.png`
- `reports/model/valid_metrics.csv`
- `reports/model/curve_fit_*C.png`
- `reports/model/prediction_scatter.png`
- `reports/model/abs_error_by_c_rate.png`
- `reports/index.html`
- `runs/dnn_best.pt`
- `runs/cnn1d_best.pt`
- `runs/fit_logs/metrics.csv` 或 TensorBoard event 文件

如果安装了 TensorBoard：

```powershell
pip install tensorboard
tensorboard --logdir runs\fit_logs
```
