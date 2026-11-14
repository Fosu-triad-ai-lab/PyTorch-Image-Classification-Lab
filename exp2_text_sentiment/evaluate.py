"""实验2：测试集评估，输出混淆矩阵、长文本/生僻词分组准确率和高置信错例。"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import LABEL_NAMES, ImdbDataset, collate, read_split, tokenize
from model import SentimentLSTM
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="IMDB 情感模型测试集评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/aclImdb")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bucket-count", type=int, default=4, help="长度/OOV 分组数")
    parser.add_argument("--max-examples", type=int, default=20, help="错例展示条数")
    parser.add_argument("--figure-dir", type=str, default="figures")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def plot_confusion_matrix(confusion, save_path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    positions = np.arange(len(LABEL_NAMES))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = confusion.max() / 2
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j, i, str(confusion[i, j]), ha="center", va="center",
                color="white" if confusion[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def bucket_accuracy(values, correct, bucket_count):
    """把样本按 values 等宽分组，返回 [(组标签, 样本数, 准确率), ...]。"""
    values = np.asarray(values)
    correct = np.asarray(correct)
    edges = np.linspace(0, values.max() + 1e-9, bucket_count + 1)
    rows = []
    for b in range(bucket_count):
        lo, hi = edges[b], edges[b + 1]
        mask = (values >= lo) & (values < hi) if b < bucket_count - 1 else (values >= lo) & (values <= hi)
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else float("nan")
        rows.append((f"{lo:.0f}~{hi:.0f}", count, acc))
    return rows


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    run_name = checkpoint_path.stem

    itos = checkpoint["itos"]
    stoi = {word: index for index, word in enumerate(itos)}
    model = SentimentLSTM(
        vocab_size=len(itos),
        embed_dim=train_args["embed_dim"],
        hidden_size=train_args["hidden_size"],
        num_layers=train_args["num_layers"],
        dropout=train_args["dropout"],
        pad_idx=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(
        f"加载模型: {checkpoint_path}（训练到第 {checkpoint['epoch']} 轮，"
        f"验证准确率 {checkpoint['val_acc']:.4f}）"
    )
    print(f"设备: {device}")

    data_dir = resolve_path(args.data_dir)
    test_texts, test_labels = read_split(data_dir, "test")
    dataset = ImdbDataset(test_texts, test_labels, stoi, train_args["max_len"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((2, 2), dtype=np.int64)
    probs_all, preds_all, correct_all = [], [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for ids, lengths, labels in tqdm(loader, desc="测试", leave=False, dynamic_ncols=True):
            ids = ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(ids, lengths)
            total_loss += criterion(outputs, labels).item() * ids.size(0)
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1).max(dim=1).values
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                confusion[t, p] += 1
            probs_all.extend(probs.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            correct_all.extend((preds == labels).cpu().tolist())
            total += ids.size(0)

    test_acc = np.trace(confusion) / confusion.sum()
    test_loss = total_loss / total

    # 长文本 / 生僻词分析：按原始分词的长度和 <unk> 占比分组（不受截断影响）
    full_lengths = np.array([len(tokenize(t)) for t in test_texts])
    oov_counts = np.array(
        [sum(1 for w in tokenize(t) if w not in stoi) for t in test_texts]
    )
    oov_rates = oov_counts / np.maximum(full_lengths, 1)

    length_rows = bucket_accuracy(full_lengths, correct_all, args.bucket_count)
    oov_rows = bucket_accuracy(oov_rates * 100, correct_all, args.bucket_count)

    # 高置信错例：模型越自信却分错的样本越值得分析
    wrong_idx = [i for i, ok in enumerate(correct_all) if not ok]
    wrong_idx.sort(key=lambda i: probs_all[i], reverse=True)
    examples = wrong_idx[: args.max_examples]

    lines = [
        f"模型: {checkpoint_path}",
        f"训练轮数: {checkpoint['epoch']}，训练时验证准确率: {checkpoint['val_acc']:.4f}",
        f"测试集样本数: {total}",
        f"测试损失: {test_loss:.4f}",
        f"测试准确率: {test_acc:.4f}",
        "",
        "各类别准确率:",
    ]
    lines += [
        f"{LABEL_NAMES[i]}: {confusion[i, i] / confusion[i].sum():.4f}" for i in range(2)
    ]
    lines += ["", "混淆矩阵（行 = 真实类别，列 = 预测类别）:", "            " + "  ".join(f"{n:>9s}" for n in LABEL_NAMES)]
    for i, name in enumerate(LABEL_NAMES):
        lines.append(f"{name:<11s} " + "  ".join(f"{confusion[i, j]:>9d}" for j in range(2)))

    lines += ["", "按评论长度分组的准确率（分词后的原始词数，不受截断影响）:"]
    lines += [f"  {name} 词: {count} 条, acc {acc:.4f}" for name, count, acc in length_rows]
    lines += ["", "按 <unk>（生僻词/未登录词）占比分组的准确率:"]
    lines += [f"  {name}%: {count} 条, acc {acc:.4f}" for name, count, acc in oov_rows]

    lines += ["", f"高置信错例 Top {len(examples)}（真实 -> 预测, 置信度）:"]
    for i in examples:
        snippet = " ".join(test_texts[i][:200].split())
        lines.append(
            f"  {LABEL_NAMES[test_labels[i]]} -> {LABEL_NAMES[preds_all[i]]}, "
            f"置信度 {probs_all[i]:.4f} | {snippet}"
        )

    log_dir, figure_dir = resolve_path(args.log_dir), resolve_path(args.figure_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    report_path = log_dir / f"{run_name}_test.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cm_path = figure_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(confusion, cm_path, f"{run_name} - Confusion matrix")

    csv_path = log_dir / f"{run_name}_buckets.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "bucket", "count", "accuracy"])
        for name, count, acc in length_rows:
            writer.writerow(["length", name, count, f"{acc:.4f}"])
        for name, count, acc in oov_rows:
            writer.writerow(["oov_rate", name, count, f"{acc:.4f}"])

    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"评估报告: {report_path}")
    print(f"混淆矩阵: {cm_path}")
    print(f"分组准确率: {csv_path}")


if __name__ == "__main__":
    main()
