# PythonAssignment - EMNIST Letter Recognition

一个基于 PyTorch 的手写字母识别系统，使用 EMNIST 数据集训练 CNN 模型，提供训练、评估、可视化和 GUI 实时识别功能。

## 功能

- ✅ 使用 EMNIST letters 数据集训练 SimpleCNN 模型
- ✅ 支持数据增强（RandomAffine）和学习率调度
- ✅ 支持可选的高斯噪声与椒盐噪声数据增强开关
- ✅ 训练过程记录 loss，并自动保存 loss 曲线图
- ✅ Tkinter GUI 实时识别手写字母
- ✅ 样本可视化（典型样本和噪声样本，输出带时间戳）
- ✅ 模型评估与准确率统计

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/jixuanzi666/PythonAssignment.git
cd PythonAssignment
```

### 2. 创建虚拟环境
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate    # Linux/Mac
```

### 3. 安装依赖
```bash
pip install torch torchvision matplotlib pillow
```

### 4. 运行评估（自动训练+测试）
```bash
python evaluate.py
# 强制重新训练：python evaluate.py --retrain
```

训练完成后会在 `outputs/` 下保存 loss 曲线图，文件名类似 `loss_curve_YYYYMMDD_HHMMSS.png`。

## 使用

### 启动 GUI 识别
```bash
python gui.py
```
在黑色画布上写大写字母，点"识别"按钮即可看到预测结果和前三候选。

### 生成样本可视化
```bash
python visualize.py
```
生成 `outputs/typical_samples_YYYYMMDD_HHMMSS.png` 和 `outputs/noisy_samples_YYYYMMDD_HHMMSS.png`。

## 文件说明

| 文件 | 用途 |
|------|------|
| `cnn_model.py` | SimpleCNN 模型定义 |
| `data_loader.py` | EMNIST 数据加载与预处理，支持可选噪声增强 |
| `train.py` | 模型训练逻辑、loss 记录与曲线可视化 |
| `evaluate.py` | 模型评估与准确率计算 |
| `gui.py` | Tkinter 图形界面 |
| `visualize.py` | 样本和噪声可视化，输出带时间戳 |

## 项目结构

```
PythonAssignment/
├── cnn_model.py
├── data_loader.py
├── train.py
├── evaluate.py
├── gui.py
├── visualize.py
├── checkpoints/          # 模型权重（git 忽略）
│   └── simplecnn_emnist.pt
├── data/                 # 数据集（git 忽略）
│   └── EMNIST/
├── outputs/              # 生成的图像（样本图、噪声图、loss 曲线）
└── README.md
```

## 本次更新说明

- `data_loader.py` 新增高斯噪声与椒盐噪声增强类，默认关闭，需要时可显式开启。
- `train.py` 记录每轮 loss，并将 loss 曲线保存到 `outputs/`。
- `visualize.py` 生成的图片文件名加入时间戳，避免重复运行时覆盖旧图。
- `evaluate.py` 仍会优先加载已有模型；若模型不存在，可通过 `--retrain` 重新训练。

## 依赖

- Python 3.8+
- torch, torchvision
- matplotlib, pillow
- tkinter（Python 标准库）

## 许可证

MIT
