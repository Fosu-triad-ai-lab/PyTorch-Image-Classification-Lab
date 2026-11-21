# 实验3：图像与文本多模态检索

## 实验内容

使用小型图文配对数据集 Flickr8k，完成图像与文本两路预处理；
调用预训练 ResNet18 提取图像特征，调用预训练 BERT-tiny 提取文本特征；
在共享嵌入空间训练图文匹配模型，并用测试集统计图像检索文本的匹配准确率；
分析模态异构带来的检索误差，对比不同嵌入维度和损失函数的影响。

## 数据集：Flickr8k

- 8091 张日常场景图片，每张 5 条英文描述；训练/验证/测试划分为 6000/1000/1000 张。
- 划分内共 8000 张图、40000 条描述；另有 91 张真实图片不在官方划分中（455 条描述），不参与训练和评估。
- 原始 `Flickr8k.token.txt` 里有一个坏文件名 `2258277193_586949ec62.jpg.1`，磁盘上没有对应图片，也不在任何划分中，读取时跳过。
- 全部 8091 张图片可正常打开；图像尺寸有 597 种，最常见的是 500×333（1523 张）和 500×375（1299 张）。
- 描述按 `[a-z0-9']+` 切词后平均 10.8 个词，中位数 10，P95 18，最长 37（按划分内 40000 条统计）；全部 8091 张图的描述词表 8493。BERT 最大长度设为 48 个 token。

数据目录（不进 Git）：

```text
data/
├─ Flicker8k_Dataset/          # 8091 张 jpg（原始目录名少一个 r）
└─ Flickr8k_text/
   ├─ Flickr8k.token.txt       # image#i \t caption
   ├─ Flickr_8k.trainImages.txt
   ├─ Flickr_8k.devImages.txt
   └─ Flickr_8k.testImages.txt
```

获取方式（节点上已存在，其他机器可任选其一）：

```bash
# 方式1：GitHub release（jbrownlee/Datasets）
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip Flickr8k_Dataset.zip
unzip Flickr8k_text.zip
```

也可从 Flickr8k 原始申请页（`https://forms.illinois.edu/sec/1713398`）或 Kaggle 的
`adityajn105/flickr8k` 获取；注意 Kaggle 版本的目录结构不同，需要整理成上面的结构。
数据引用：Hodosh, Young, Hockenmaier. *Framing Image Description as a Ranking Task*. JAIR 2013.

## 环境与预训练模型

在 DGX Spark 节点上：

```bash
source ~/venvs/cnn/bin/activate
```

共享环境：Python 3.12、torch 2.14.1+cu130、torchvision 0.29.1+cu130、
matplotlib、tqdm，以及本实验新增的 `transformers 4.57.6`。
ResNet18 的 `IMAGENET1K_V1` 权重已缓存在 `~/.cache/torch/hub/checkpoints/`。

文本模型使用官方 tiny BERT：`google/bert_uncased_L-2_H-128_A-2`
（4.39M 参数，2 层，hidden 128）。已下载到 `data/models/bert-tiny/`，
该目录不进 Git。其他机器可直连 hf-mirror 下载：

```bash
mkdir -p data/models/bert-tiny
cd data/models/bert-tiny
for f in config.json vocab.txt model.safetensors; do
  curl -L --fail -O "https://hf-mirror.com/google/bert_uncased_L-2_H-128_A-2/resolve/main/$f"
done
cd ../../..
```

共享 venv 缺包时不要自己 `pip install`，在群里找 Lucky 统一安装（同 `docs/复刻指南.md` 第三节）。

## 模型方案

```text
图像: 冻结 ResNet18 (512d) -> MLP -> 256d -> L2 normalize
文本: 预训练 BERT-tiny (mean pooling, 128d) -> MLP -> 256d -> L2 normalize
相似度: 余弦相似度；损失 InfoNCE（默认）或 batch-hard 三元组损失
```

- 图像特征在训练前一次性算好，ResNet18 不参与反向传播；这符合大纲“调用预训练 CNN 提取特征”的要求，也让特征维度和损失函数对比更便宜。
- 文本编码器默认微调 BERT-tiny；加 `--freeze-text` 可只训练投影头，作为对比项。
- 训练时每个 epoch 为每张图随机抽 1 条描述，batch 内其他图的描述作为负样本，避免同一张图的 5 条描述互相被当成负样本。
- 检索评估时使用测试集全部 5 条描述：图像→文本统计 Recall@1/5/10（任意一条正确描述进入 top-K 即命中）和中位排名；同时给出文本→图像的对应指标。
- 固定随机种子 42。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `data.py` | Flickr8k 解析、划分校验、数据审计、图像预处理、特征缓存加载 |
| `precompute_features.py` | 用冻结 ResNet18 预计算 train/dev/test 图像特征 |
| `model.py` | 图像/文本投影头、BERT 文本编码器、InfoNCE 与三元组损失 |
| `encoding.py` | 批量编码描述文本（训练和评估共用） |
| `retrieval.py` | Recall@K、中位排名 |
| `train.py` | 训练入口：argparse、固定种子、每轮验证、保存最优模型、画曲线 |
| `evaluate.py` | 测试集检索指标、定性示例 CSV/图片 |
| `rank_stats.py` | 真值排名分布、失败样本统计（报告 8.1 数字来源） |
| `data/` | Flickr8k、BERT-tiny、ResNet18 特征缓存（不进 Git） |
| `figures/` | 训练曲线、检索示例图 |
| `logs/` | CSV、配置、测试报告、检索示例、训练耗时 |
| `checkpoints/` | 模型权重（不进 Git） |

## 怎么跑

在 `exp3_image_text_retrieval/` 目录下：

```bash
source ~/venvs/cnn/bin/activate

# 1. 预计算冻结 ResNet18 特征（只需一次）
python precompute_features.py

# 2. 训练基线（10 epoch，GB10 上约 11 秒）
python train.py --loss infonce --embed-dim 256 --epochs 10 \
  --run-name baseline-infonce-d256

# 3. 测试集评估
python evaluate.py --checkpoint checkpoints/baseline-infonce-d256.pth --split test

# 4. 维度 / 损失对比（其余 7 组完整命令，run 名与「结果记录」表一致）
python train.py --loss infonce --embed-dim 128 --epochs 10 --run-name infonce-d128
python train.py --loss infonce --embed-dim 512 --epochs 10 --run-name infonce-d512
python train.py --loss infonce --embed-dim 512 --epochs 30 --run-name infonce-d512-ep30
python train.py --loss infonce --embed-dim 512 --epochs 10 --freeze-text --run-name infonce-d512-frozen
python train.py --loss triplet --embed-dim 256 --epochs 10 --run-name triplet-d256
python train.py --loss triplet --embed-dim 512 --epochs 10 --run-name triplet-d512
# 诊断组：学习率 ×5 + 30 epoch，用于确认 triplet 不收敛并非学习率问题
python train.py --loss triplet --embed-dim 512 --epochs 30 --lr 5e-3 --text-lr 5e-4 --run-name triplet-d512-lrx5

# 每组训练完都跑一次测试集评估（替换 --checkpoint 即可）
python evaluate.py --checkpoint checkpoints/<实验名>.pth --split test
```

常用参数：`--batch-size 128`、`--text-lr 1e-4`、`--lr 1e-3`、`--max-len 48`、
`--freeze-text`、`--device cpu`。

## 前置准备与 smoke test

- 数据完整性：8091 张图全部可读；官方划分内 8000 张图全部有 5 条描述；坏文件名已跳过。
- 特征预计算：train 6000 张 5.6 秒，dev/test 各 1000 张 1.0 秒，输出 512 维 float32 特征。
- 单 epoch smoke test 通过：训练 1.4 秒，峰值显存 0.23 GB，说明数据、BERT、损失和检索评估链路可用。
- `--loss triplet` 和 `--freeze-text` 两条分支也各跑过 1 epoch smoke，训练和验证均正常。

## 结果记录

对比实验均已跑完（除基线外均为 2026-11-19 补充；triplet 的 lr×5 组是诊断性重跑）。dev 列为最优轮（按 dev I2T R@1 选）：

| 实验 | dev I2T R@1 | dev I2T R@5 | dev I2T R@10 | test I2T R@1/5/10 | test T2I R@1/5/10 | 训练耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline-infonce-d256`（10 epoch） | 13.80%（epoch 8） | 30.70% | 44.10% | 12.30% / 33.30% / 47.70% | 10.18% / 30.06% / 43.78% | 10.6 秒 |
| `infonce-d128`（10 epoch） | 11.80%（epoch 9） | 30.00% | 42.30% | 11.30% / 32.10% / 45.40% | 9.08% / 28.42% / 41.22% | 9.2 秒 |
| `infonce-d512`（10 epoch） | 14.40%（epoch 10） | 35.30% | 47.90% | 16.20% / 35.80% / 48.10% | 11.42% / 33.36% / 46.64% | 12.7 秒 |
| `infonce-d512-ep30`（30 epoch） | 17.00%（epoch 14） | 39.80% | 51.40% | 17.70% / 38.10% / 51.30% | 12.00% / 34.40% / 48.24% | 28.4 秒 |
| `infonce-d512-frozen`（10 epoch） | 10.70%（epoch 8） | 28.70% | 41.10% | 9.60% / 29.60% / 40.50% | 7.96% / 24.42% / 36.52% | 9.5 秒 |
| `triplet-d256`（10 epoch） | 0.40%（epoch 6） | 0.90% | 1.80% | 0.30% / 0.90% / 1.50% | 0.10% / 0.66% / 1.20% | 9.6 秒 |
| `triplet-d512`（10 epoch） | 0.20%（epoch 8） | 0.50% | 1.20% | 0.10% / 0.50% / 1.60% | 0.10% / 0.66% / 1.12% | 9.7 秒 |
| `triplet-d512-lrx5`（30 epoch，lr×5） | 0.30%（epoch 5） | 0.40% | 0.70% | 0.20% / 0.70% / 0.80% | 0.10% / 0.52% / 1.00% | 27.8 秒 |

主要结论：

- **嵌入维度**：同口径（InfoNCE、10 epoch）下 d512 > d256 > d128；d512 在 10 epoch 时最优轮是第 10 轮、尚未到平台，加长到 30 epoch 后 test R@1 从 16.20% 升到 17.70%（最优轮 14）。每 epoch 约 1 秒，加长成本很低。
- **损失函数**：batch-hard 三元组损失（余弦相似度 + margin 0.2）在本设置下无法收敛——d256/d512/lr×5+30 epoch 三组 R@1 全部 ≤ 0.4%（与随机检索相当，中位排名 550–800），train_loss 长期卡在 margin 尺度附近；而 InfoNCE 带 temperature 0.07 缩放，正常收敛。说明温度缩放在该任务上不可省。
- **冻结 vs 微调 BERT**：同口径（d512、10 epoch）下微调 16.20% 对冻结 9.60%，小数据下微调文本编码器帮助明显。
- 基线（10 epoch）测试集图像→文本中位排名 12，文本→图像中位排名 14；30 epoch d512 分别为 10 和 11。

## 计划实验

1. ~~嵌入维度：128 / 256 / 512，固定 InfoNCE 和 10 epoch，比较 R@1/5/10。~~ 已跑（含 30 epoch 收敛参考）。
2. ~~损失函数：InfoNCE vs batch-hard 三元组，固定最佳维度。~~ 已跑；triplet 各组均未收敛，另补 lr×5 诊断组确认非学习率问题。
3. ~~可选：冻结 BERT vs 微调 BERT，观察小数据下预训练特征的适配程度。~~ 已跑。
4. ~~误差分析：从 `logs/<run>_test_examples.csv` 和检索示例图挑失败样本，分析模态异构、细粒度物体、计数/关系描述等错误原因。~~ 已完成，见 `docs/实验3报告.md` 第八节；排名分布等定量统计由 `rank_stats.py` 生成（`logs/infonce-d512-ep30_rank_stats.txt`）。

## 复现说明

- 固定随机种子 42（Python、NumPy、PyTorch、CUDA），开启 `cudnn.deterministic`。
- 数据划分和描述采样由种子决定；同一节点上重跑结果应一致。
- 数据和模型权重不进 Git；`data/features/`、`data/models/`、`checkpoints/` 需要按上面的命令重新生成/下载。
- 日志和图表保留在 `logs/`、`figures/`，便于核对报告数字。
