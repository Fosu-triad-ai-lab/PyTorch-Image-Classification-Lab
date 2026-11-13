"""实验1：在测试集上评估模型，输出混淆矩阵、分类报告和错例图。"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, build_transforms
from model import LightweightCNN

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 模型测试集评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--figure-dir", type=str, default="figures")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--max-mistakes", type=int, default=16, help="错例图最多展示几张")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def plot_confusion_matrix(confusion, save_path, title):
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    positions = np.arange(len(CLASS_NAMES))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = confusion.max() / 2
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if confusion[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mistakes(mistakes, save_path, columns=8):
    count = len(mistakes)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.7, rows * 2.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    mean = np.array(CIFAR10_MEAN)
    std = np.array(CIFAR10_STD)
    for ax, (image, true_label, pred_label) in zip(axes, mistakes):
        image = image.permute(1, 2, 0).numpy() * std + mean
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(
            f"true: {CLASS_NAMES[true_label]}\npred: {CLASS_NAMES[pred_label]}",
            fontsize=7,
            color="red",
        )
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = resolve_path(args.checkpoint)
    data_dir = resolve_path(args.data_dir)
    figure_dir = resolve_path(args.figure_dir)
    log_dir = resolve_path(args.log_dir)
    for directory in (figure_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    train_args = checkpoint["args"]
    run_name = checkpoint_path.stem

    model = LightweightCNN(
        num_classes=len(CLASS_NAMES),
        use_bn=train_args["use_bn"],
        use_dropout=train_args["use_dropout"],
        dropout_p=train_args["dropout_p"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(
        f"加载模型: {checkpoint_path}（训练到第 {checkpoint['epoch']} 轮，"
        f"验证准确率 {checkpoint['val_acc']:.4f}）"
    )
    print(f"设备: {device}")

    _, eval_transform = build_transforms()
    test_set = CIFAR10(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    criterion = nn.CrossEntropyLoss()
    num_classes = len(CLASS_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    mistakes = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader, desc="测试", leave=False, dynamic_ncols=True
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            for true_label, pred_label in zip(
                labels.cpu().tolist(), predictions.cpu().tolist()
            ):
                confusion[true_label, pred_label] += 1

            wrong_index = (predictions != labels).nonzero(as_tuple=True)[0]
            for index in wrong_index[: max(0, args.max_mistakes - len(mistakes))]:
                mistakes.append(
                    (
                        images[index].cpu(),
                        labels[index].item(),
                        predictions[index].item(),
                    )
                )
                if len(mistakes) >= args.max_mistakes:
                    break

    test_loss = total_loss / total
    test_acc = np.trace(confusion) / confusion.sum()
    per_class_acc = confusion.diagonal() / confusion.sum(axis=1)

    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    confused_pairs = sorted(
        (
            (off_diagonal[i, j], i, j)
            for i in range(num_classes)
            for j in range(num_classes)
            if i != j
        ),
        reverse=True,
    )

    matrix_lines = []
    for row in confusion:
        matrix_lines.append(" ".join(f"{value:5d}" for value in row))

    report_lines = [
        f"模型: {checkpoint_path}",
        f"训练轮数: {checkpoint['epoch']}",
        f"训练时验证准确率: {checkpoint['val_acc']:.4f}",
        f"测试集样本数: {total}",
        f"测试损失: {test_loss:.4f}",
        f"测试准确率: {test_acc:.4f}",
        "",
        "各类别准确率:",
    ]
    report_lines += [
        f"{name}: {accuracy:.4f}" for name, accuracy in zip(CLASS_NAMES, per_class_acc)
    ]
    report_lines += ["", "最容易混淆的类别对（真实 -> 预测: 数量）:"]
    report_lines += [
        f"{CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: {count}"
        for count, i, j in confused_pairs[:5]
    ]
    report_lines += [
        "",
        "混淆矩阵（行 = 真实类别，列 = 预测类别）:",
        "       " + " ".join(f"{name[:5]:>5s}" for name in CLASS_NAMES),
    ]
    for name, line in zip(CLASS_NAMES, matrix_lines):
        report_lines.append(f"{name[:6]:<6s} {line}")

    report_path = log_dir / f"{run_name}_test.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    cm_path = figure_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(confusion, cm_path, f"{run_name} - Confusion matrix")

    mistake_path = figure_dir / f"{run_name}_misclassified.png"
    if mistakes:
        plot_mistakes(mistakes, mistake_path)

    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"评估报告: {report_path}")
    print(f"混淆矩阵: {cm_path}")
    if mistakes:
        print(f"错例图: {mistake_path}")


if __name__ == "__main__":
    main()
