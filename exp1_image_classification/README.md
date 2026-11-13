# 实验1：基于 PyTorch 的图像分类

## 实验内容

基于 CIFAR-10 训练轻量卷积神经网络，完成数据加载、增强、训练、验证和测试全流程；
绘制 loss/accuracy 曲线和混淆矩阵，分析错例；对比 BatchNorm、Dropout 对模型性能的影响。

## 数据集

- **CIFAR-10**：10 类彩色图像（airplane/automobile/bird/cat/deer/dog/frog/horse/ship/truck），32x32，共 60000 张。
- 划分：官方训练集 50000 张按 9:1 划分为训练集 45000 张、验证集 5000 张；官方测试集 10000 张只用于最终评估。
- 预处理：训练集做 `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` + 归一化；验证/测试集只做归一化和张量化。
- 数据放在 `data/`，首次运行由 `torchvision` 自动下载（约 170MB），该目录不进 Git。

## 环境

在 DGX Spark 节点上：

```bash
source ~/venvs/cnn/bin/activate
```

共享环境已包含 Python 3.12.3、torch 2.14.1+cu130、torchvision 0.29.1+cu130、matplotlib、tqdm。
GPU 为 NVIDIA GB10，代码会自动选择 CUDA；CPU 仅用于跑通流程。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `data.py` | CIFAR-10 下载、训练/验证/测试划分、数据增强 |
| `model.py` | 轻量 CNN（3 个卷积块 + 1 个隐藏全连接层），BN/Dropout 可开关 |
| `train.py` | 训练入口：argparse、固定种子、训练循环、保存最优模型、画曲线 |
| `evaluate.py` | 测试集评估：准确率、各类别准确率、混淆矩阵、错例图 |
| `logs/` | 每组实验的 CSV 日志、配置 JSON、测试报告 |
| `figures/` | 曲线、混淆矩阵、错例图 |
| `checkpoints/` | 模型权重，不进 Git |

## 怎么跑

以下命令在 `exp1_image_classification/` 目录下执行。正式训练放到 tmux 里：

```bash
source ~/venvs/cnn/bin/activate
cd ~/cnn/lucky/lab/exp1_image_classification
tmux new -s cnn-lucky
```

### 1. 训练（4 组对比实验）

```bash
# 基准：无 BN、无 Dropout
python train.py

# 仅加 BatchNorm
python train.py --use-bn

# 仅加 Dropout
python train.py --use-dropout

# BatchNorm + Dropout
python train.py --use-bn --use-dropout
```

每组实验按配置自动命名，结果互不覆盖：

| 实验名 | BN | Dropout |
| --- | --- | --- |
| `simplecnn_bn0_drop0` | 否 | 否 |
| `simplecnn_bn1_drop0` | 是 | 否 |
| `simplecnn_bn0_drop1` | 否 | 是 |
| `simplecnn_bn1_drop1` | 是 | 是 |

### 2. 测试集评估

```bash
python evaluate.py --checkpoint checkpoints/simplecnn_bn1_drop1.pth
```

对四组模型分别执行，替换 `--checkpoint` 即可。

### 3. 常用参数

```bash
python train.py --epochs 1 --batch-size 64   # 先小规模确认流程
python train.py --device cpu                 # 强制用 CPU（很慢，仅验证代码）
```

## 默认超参数

| 参数 | 默认值 |
| --- | --- |
| 随机种子 | 42 |
| 训练轮数 | 20 |
| batch size | 128 |
| 优化器 | SGD（momentum 0.9，weight decay 5e-4） |
| 初始学习率 | 0.05，余弦退火 |
| 损失函数 | CrossEntropyLoss |
| Dropout 概率 | 0.3（卷积块后和全连接层前） |
| 最优模型 | 按验证集准确率保存 |

## 输出

- `logs/simplecnn_bnX_dropX.csv`：每轮训练/验证 loss、准确率、学习率
- `logs/simplecnn_bnX_dropX_config.json`：本次运行的完整参数
- `figures/simplecnn_bnX_dropX_curves.png`：loss 和 accuracy 曲线
- `checkpoints/simplecnn_bnX_dropX.pth`：验证集上最好的模型（不进 Git）
- `evaluate.py` 额外产出：
  - `logs/simplecnn_bnX_dropX_test.txt`：测试准确率、各类别准确率、最易混淆类别对、混淆矩阵
  - `figures/simplecnn_bnX_dropX_confusion_matrix.png`
  - `figures/simplecnn_bnX_dropX_misclassified.png`：分错样本示例

图中类别名用英文，避免节点缺少中文字体导致乱码；对应中文为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

## 结果记录

| 实验名 | 训练轮数 | 最好验证准确率 | 测试准确率 | 训练耗时 |
| --- | --- | --- | --- | --- |
| `simplecnn_bn0_drop0` | 20 | 88.14% | 87.82% | 2 分 01 秒 |
| `simplecnn_bn1_drop0` | 20 | **90.68%** | **90.21%** | 2 分 16 秒 |
| `simplecnn_bn0_drop1` | 20 | 86.52% | 86.65% | 2 分 02 秒 |
| `simplecnn_bn1_drop1` | 20 | 88.62% | 88.13% | 2 分 19 秒 |

GB10 上四组 20 epoch 实测每组 2 分 01 秒～2 分 19 秒，四组训练加评估共约 9 分钟。
以 `logs/` 和 `figures/` 里的实际数字为准，完整分析见 `docs/实验1报告.md`。

## 复现说明

- 固定随机种子 42（Python、NumPy、PyTorch、CUDA），开启 `cudnn.deterministic`。
- 数据和模型全部在节点本地，重新运行同样的命令即可复现。
- 同一配置重复运行时，结果允许有小的数值波动，以日志为准。
