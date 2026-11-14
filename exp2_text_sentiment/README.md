# 实验2：基于 PyTorch 的文本情感分类

## 实验内容

在 IMDB 影评数据集上训练 Embedding + LSTM 模型完成情感二分类；
完成文本清洗、分词、词表构建和序列数字化；
对比随机初始化 Embedding 与 GloVe 预训练词向量（大纲拓展项）；
按评论长度和生僻词占比分组分析错误原因；
额外做噪声扰动实验（替换/删词）量化模型的鲁棒性。

## 数据集

- **aclImdb_v1**：IMDB 影评情感数据集，训练/测试各 25000 条（正负各半），
  解压后 `data/aclImdb/{train,test}/{pos,neg}/` 各 12500 个 txt 文件。
- 训练集内部按 9:1 切成训练 22500 条、验证 2500 条（固定随机流），官方测试集 25000 条只做最终评估。
- 预处理：去 HTML 标签 → 小写化 → 正则 `[a-z0-9']+` 分词 → 词表 2 万（词频 ≥ 2）→ 超长截断 256 词。
- 数据放在 `data/`，不进 Git。

### 下载方法（国内网络实测可用）

```bash
mkdir -p data/glove && cd data
# 官方 aclImdb 原包（84MB，字节数与斯坦福官方一致）
wget -c "https://hf-mirror.com/datasets/NolanChai/aclImdb_v1/resolve/main/aclImdb_v1.tar.gz"
tar -xzf aclImdb_v1.tar.gz
# GloVe 6B 词向量（862MB zip，只抽 100d）
wget -c "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip"
unzip -j -o glove.6B.zip "glove.6B.100d.txt" -d glove/ && rm glove.6B.zip
```

备用源：魔搭 `lemon3344s/imdb-fdu-for-lecture-2510`（`train.txt`/`test.txt` 格式，
需转换脚本）；官方原始地址 `ai.stanford.edu/~amaas/data/sentiment/`（节点直连不通）。

## 环境

在 DGX Spark 节点上 `source ~/venvs/cnn/bin/activate`。
共享环境已含 Python 3.12.3、torch 2.14.1+cu130、matplotlib、tqdm、numpy，
本实验**不需要额外安装任何包**。词向量文件约 348MB，加载一次约 20 秒。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `data.py` | 清洗分词、词表构建、GloVe 矩阵加载、数据集与 DataLoader |
| `model.py` | SentimentLSTM：Embedding → Dropout → LSTM（末隐状态）→ Dropout → 全连接 |
| `train.py` | 训练入口：argparse、固定种子、训练循环、保存最优模型、画曲线 |
| `evaluate.py` | 测试集评估：准确率、混淆矩阵、长度/生僻词分组准确率、高置信错例 |
| `noise_test.py` | 噪声鲁棒性实验：替换/删词两种扰动 × 5 个强度，画准确率下降曲线 |
| `logs/` | CSV 日志、配置 JSON、测试报告、分组准确率 |
| `figures/` | 曲线、混淆矩阵、噪声曲线 |
| `checkpoints/` | 模型权重，不进 Git |

## 怎么跑

在 `exp2_text_sentiment/` 目录下：

```bash
source ~/venvs/cnn/bin/activate

# 训练（两组对比，各 10 轮，GB10 上每组约 36 秒）
python train.py --embedding-init random
python train.py --embedding-init glove

# 测试集评估（含长度/生僻词分组与错例）
python evaluate.py --checkpoint checkpoints/lstm_random.pth
python evaluate.py --checkpoint checkpoints/lstm_glove.pth

# 噪声鲁棒性实验（两模型一起）
python noise_test.py --checkpoints checkpoints/lstm_random.pth checkpoints/lstm_glove.pth
```

常用参数：`--epochs 1` 冒烟、`--vocab-size`、`--max-len`、`--batch-size`（64 起步可试 128/256）、
`--lr`、`--hidden-size`、`--device cpu`。

## 默认超参数

| 参数 | 取值 |
| --- | --- |
| 随机种子 | 42（Python、NumPy、PyTorch、CUDA） |
| 词表 | 2 万（词频 ≥ 2），`<pad>=0`、`<unk>=1` |
| 截断长度 | 256 词 |
| 词向量维度 | 100（GloVe 用 glove.6B.100d，命中即用、未命中的词用全体向量均值，训练中微调） |
| LSTM | 1 层，hidden 128，取最后时刻隐状态 |
| Dropout | 0.3（Embedding 输出后、分类器前各一次） |
| 优化器 | Adam，lr 1e-3，weight decay 1e-4，梯度裁剪 5.0 |
| batch size | 64 |
| 训练轮数 | 10，按验证集准确率保存最优 |

## 输出

- `logs/lstm_{random,glove}.csv`：每轮训练/验证 loss、准确率
- `logs/lstm_{random,glove}_config.json`：本次运行完整参数（含 GloVe 命中率）
- `logs/lstm_{random,glove}_test.txt`：测试报告、混淆矩阵、长度/生僻词分组、高置信错例
- `logs/lstm_{random,glove}_buckets.csv`：分组准确率
- `logs/noise_results.csv`：噪声实验全部数字
- `figures/lstm_{random,glove}_curves.png`：loss/accuracy 曲线
- `figures/lstm_{random,glove}_confusion_matrix.png`：混淆矩阵
- `figures/noise_curves.png`：噪声强度 - 准确率曲线
- `checkpoints/lstm_{random,glove}.pth`：验证集最优模型（不进 Git）

## 结果记录

### 主对比（随机初始化 vs 预训练词向量）

| 实验 | 最好验证准确率 | 最好轮次 | 测试准确率 | 测试损失 | 训练耗时 |
| --- | --- | --- | --- | --- | --- |
| `lstm_random` | **88.32%** | 9 | **87.73%** | 0.2999 | 0.6 分钟 |
| `lstm_glove` | 87.96% | 9 | 85.85% | 0.4765 | 0.6 分钟 |

GloVe 词表命中率 95.75%（词表 19998 个实词命中 19150 个）。
每组训练 GB10 上 0.6 分钟，全部实验（两组训练 + 评估 + 噪声）合计约 4 分钟。

### 噪声鲁棒性（测试集扰动后的准确率）

| 噪声强度 p | 删词-random | 删词-glove | 替换-random | 替换-glove |
| --- | --- | --- | --- | --- |
| 0.0 | 87.73% | 85.85% | 87.73% | 85.85% |
| 0.1 | 86.92% | 85.02% | 86.63% | 83.24% |
| 0.2 | 86.00% | 83.98% | 85.06% | 80.14% |
| 0.3 | 84.72% | 82.82% | 82.60% | 76.93% |
| 0.4 | 83.54% | 81.87% | 79.42% | 73.03% |

以 `logs/` 和 `figures/` 里的实际数字为准；详细分析见 `docs/实验2报告.md`（待写）。

## 复现说明

- 固定随机种子 42（Python、NumPy、PyTorch、CUDA），开启 `cudnn.deterministic`；
  验证集划分、词表顺序、噪声扰动全部由该种子派生，重跑结果一致。
- 同一配置重复运行允许小幅数值波动，以日志为准。
