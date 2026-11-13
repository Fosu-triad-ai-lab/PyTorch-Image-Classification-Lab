"""实验1：CIFAR-10 图像分类训练入口。"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import CLASS_NAMES, build_dataloaders
from model import LightweightCNN

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 轻量 CNN 训练")
    parser.add_argument("--data-dir", type=str, default="data", help="CIFAR-10 数据目录")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-bn", action="store_true", help="加入 BatchNorm")
    parser.add_argument("--use-dropout", action="store_true", help="加入 Dropout")
    parser.add_argument("--dropout-p", type=float, default=0.3)
    parser.add_argument(
        "--run-name", type=str, default="", help="实验名，默认按 BN/Dropout 开关生成"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--figure-dir", type=str, default="figures")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(
        loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False, dynamic_ncols=True
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
        progress.set_postfix(
            loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}"
        )
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
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

    run_name = args.run_name or (
        f"simplecnn_bn{int(args.use_bn)}_drop{int(args.use_dropout)}"
    )
    data_dir = resolve_path(args.data_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    log_dir = resolve_path(args.log_dir)
    figure_dir = resolve_path(args.figure_dir)
    for directory in (checkpoint_dir, log_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"实验名: {run_name}")
    print(f"设备: {device}")
    print(
        f"配置: BN={args.use_bn} Dropout={args.use_dropout} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}"
    )

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = LightweightCNN(
        num_classes=len(CLASS_NAMES),
        use_bn=args.use_bn,
        use_dropout=args.use_dropout,
        dropout_p=args.dropout_p,
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    csv_path = log_dir / f"{run_name}.csv"
    config_path = log_dir / f"{run_name}_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), "run_name": run_name, "device": str(device)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            lr = optimizer.param_groups[0]["lr"]

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
                    f"{lr:.6f}",
                ]
            )
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.5f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "class_names": list(CLASS_NAMES),
                    },
                    checkpoint_path,
                )
                print(f"  -> 保存最优模型（val acc {best_val_acc:.4f}）")

            scheduler.step()

    curve_path = figure_dir / f"{run_name}_curves.png"
    plot_curves(history, curve_path, run_name)

    print(f"\n训练结束，最优验证准确率: {best_val_acc:.4f}")
    print(f"日志: {csv_path}")
    print(f"曲线: {curve_path}")
    print(f"最优模型: {checkpoint_path}")


if __name__ == "__main__":
    main()
