"""实验2：IMDB 情感分类训练入口（随机初始化 / GloVe 预训练词向量对比）。"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import LABEL_NAMES, build_dataloaders, build_vocab, load_glove_matrix
from model import SentimentLSTM

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="IMDB Embedding+LSTM 情感分类训练")
    parser.add_argument("--data-dir", type=str, default="data/aclImdb")
    parser.add_argument("--glove-path", type=str, default="data/glove/glove.6B.100d.txt")
    parser.add_argument(
        "--embedding-init",
        type=str,
        default="random",
        choices=["random", "glove"],
        help="词向量初始化方式（大纲拓展对比项）",
    )
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="", help="默认按词向量初始化方式命名")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--figure-dir", type=str, default="figures")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    progress = tqdm(
        loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False, dynamic_ncols=True
    )
    for ids, lengths, labels in progress:
        ids = ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(ids, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪，防止 LSTM 偶发的梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * ids.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += ids.size(0)
        progress.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for ids, lengths, labels in loader:
        ids = ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(ids, lengths)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * ids.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += ids.size(0)
    return total_loss / total, correct / total


def plot_curves(history, save_path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="val", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="val", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_name = args.run_name or f"lstm_{args.embedding_init}"
    checkpoint_dir, log_dir, figure_dir = (
        resolve_path(args.checkpoint_dir),
        resolve_path(args.log_dir),
        resolve_path(args.figure_dir),
    )
    for directory in (checkpoint_dir, log_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"实验名: {run_name}")
    print(f"设备: {device}")

    # 1. 词表：只用训练集文本构建，验证/测试集一律通过 <unk> 映射
    train_texts_all, train_labels_all = None, None
    data_dir = resolve_path(args.data_dir)
    from data import read_split

    train_texts_all, train_labels_all = read_split(data_dir, "train")
    stoi, itos = build_vocab(train_texts_all, args.vocab_size, args.min_freq)
    print(f"词表大小: {len(itos)}（词频 >= {args.min_freq}，上限 {args.vocab_size}）")

    # 2. 词向量初始化
    embedding_matrix, hit_rate = None, 0.0
    if args.embedding_init == "glove":
        glove_path = resolve_path(args.glove_path)
        embedding_matrix, hit_rate = load_glove_matrix(itos, glove_path)
        print(f"GloVe 命中率: {hit_rate:.2%}（词表 {len(itos) - 2} 个实词中命中 "
              f"{round(hit_rate * (len(itos) - 2))} 个）")

    # 3. 数据加载器（训练集内部再切 9:1 训练/验证）
    train_loader, val_loader, _, _ = build_dataloaders(
        data_dir, stoi, args.batch_size, args.max_len, args.val_ratio
    )
    print(f"训练集 {len(train_loader.dataset)} 条，验证集 {len(val_loader.dataset)} 条")

    model = SentimentLSTM(
        vocab_size=len(itos),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=0,
        embedding_matrix=embedding_matrix,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    csv_path = log_dir / f"{run_name}.csv"
    config_path = log_dir / f"{run_name}_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "run_name": run_name,
                "device": str(device),
                "vocab_built": len(itos),
                "glove_hit_rate": hit_rate,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs,
                args.grad_clip,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                ]
            )
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "itos": itos,
                        "glove_hit_rate": hit_rate,
                        "class_names": list(LABEL_NAMES),
                    },
                    checkpoint_path,
                )
                print(f"  -> 保存最优模型（val acc {best_val_acc:.4f}）")

    elapsed = time.time() - start
    curve_path = figure_dir / f"{run_name}_curves.png"
    plot_curves(history, curve_path, run_name)

    print(f"\n训练结束，耗时 {elapsed / 60:.1f} 分钟，最优验证准确率: {best_val_acc:.4f}")
    print(f"日志: {csv_path}")
    print(f"曲线: {curve_path}")
    print(f"最优模型: {checkpoint_path}")


if __name__ == "__main__":
    main()
