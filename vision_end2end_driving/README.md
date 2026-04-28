# Vision End-to-End Driving Mini Project

这个项目用于学习“视觉端到端自动驾驶”的完整算法链路：数据模拟生成、数据清洗、特征/标签理解、CNN 模型训练、CUDA 加速、TensorBoard/CSV 日志、评估和结果应用。

## Pipeline

1. `src/simulate_data.py`
   生成前视摄像头道路图像，以及 `steering / throttle / brake` 控制标签。图像包含弯道、车道偏移、障碍物、雨、雾、眩光、噪声和少量坏帧。

2. `src/clean_dataset.py`
   检查图像存在性、尺寸、亮度、对比度和标签范围，输出 `data/processed/train.csv`、`val.csv`、`test.csv`。

3. `src/visualize_data.py`
   生成数据分布图、场景比例图和样例图，帮助理解数据特性。

4. `src/train.py`
   使用 CUDA 训练 DAVE-2 风格 CNN，从图像直接预测 `steering / throttle / brake`。如果安装了 TensorBoard，会写入 event；否则自动写 `runs/e2e_cnn/metrics.csv`。

5. `src/evaluate.py`
   在测试集上输出 MAE、RMSE、R2、预测散点图、误差分布和最差样例。

6. `src/explain_model.py`
   对 steering 输出生成 saliency 热力图，帮助观察模型关注的图像区域。

7. `src/make_report.py`
   生成 `reports/index.html`，把数据、模型指标和解释性结果汇总到一个页面。

## 一键运行

在本目录下执行：

```powershell
python src/run_pipeline.py --samples 3000
```

快速验证可以少生成一些数据：

```powershell
python src/run_pipeline.py --samples 600 --epochs 2
```

## TensorBoard

当前环境如果没有安装 TensorBoard，训练会自动写 CSV 日志。安装后可运行：

```powershell
pip install tensorboard
tensorboard --logdir runs/e2e_cnn
```

然后在浏览器打开 TensorBoard 给出的本地地址。

## 你应该重点观察什么

- `reports/index.html`：完整实验入口。
- `reports/data/label_distributions.png`：控制标签是否均衡，是否覆盖足够的转向和制动场景。
- `reports/data/steering_vs_curvature.png`：标签是否真的和道路几何有关。
- `runs/e2e_cnn`：训练/验证 loss 和 MAE 是否同步下降。
- `reports/model/prediction_scatter.png`：模型预测是否贴近真实控制量。
- `reports/model/worst_steering_examples.png`：失败样例通常揭示数据覆盖不足、噪声过强或模型容量不足。
- `reports/explain/saliency_*.png`：模型是否把梯度敏感区域放在车道线、道路边缘、障碍物等关键视觉区域。

## 学习扩展方向

- 数据：加入夜晚、施工区、多车道、遮挡、不同相机视角。
- 模型：尝试 ResNet backbone、时序模型 CNN+LSTM、Transformer。
- 标签：把控制回归扩展为轨迹点预测。
- 安全：加入不确定性估计、OOD 检测和规则保护层。
