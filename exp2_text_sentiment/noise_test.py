"""实验2 拓展：文本噪声鲁棒性实验。

对测试集做两种扰动并测量准确率下降曲线：
- replace：以概率 p 把词替换成词表中的随机词（模拟噪声、错别字、生僻词入侵）；
- delete ：以概率 p 随机删词（模拟信息缺失）。

两种词向量初始化的模型都跑，产出 CSV 和对比曲线图。
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import PAD_IDX, UNK_IDX, ImdbDataset, collate, read_split
from model import SentimentLSTM
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="IMDB 模型噪声鲁棒性实验")
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", required=True,
        help="一个或多个模型权重路径，每条权重画一条曲线",
    )
    parser.add_argument("--data-dir", type=str, default="data/aclImdb")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--modes", type=str, nargs="+", default=["replace", "delete"],
                        choices=["replace", "delete"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--figure-dir", type=str, default="figures")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def perturb_dataset(dataset, mode, rate, rng, vocab_size):
    """在 id 序列层面做扰动，返回新的样本列表。

    replace: 非 <pad> 的 token 以概率 p 替换为词表内随机 id（避开 <pad>/<unk>）；
    delete : token 以概率 p 删除，删除后至少保留 1 个 token。
    """
    if rate == 0.0:
        return list(dataset.samples)

    keep_probs = rng.random(len(dataset.samples))
    new_samples = []
    for (seq, label), _ in zip(dataset.samples, keep_probs):
        ids = seq.tolist()
        if mode == "replace":
            mask = rng.random(len(ids)) < rate
            for i in np.nonzero(mask)[0]:
                ids[i] = int(rng.integers(2, vocab_size))
        else:  # delete
            keep = rng.random(len(ids)) >= rate
            ids = [t for t, k in zip(ids, keep) if k] or [ids[0]]
        new_samples.append((torch.tensor(ids, dtype=torch.long), label))
    return new_samples


@torch.no_grad()
def accuracy_of(model, samples, device, batch_size):
    loader = DataLoader(
        samples, batch_size=batch_size, shuffle=False, collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    correct, total = 0, 0
    for ids, lengths, labels in tqdm(loader, leave=False, dynamic_ncols=True):
        ids = ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(ids, lengths).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += ids.size(0)
    return correct / total


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    itos = checkpoint["itos"]
    stoi = {word: index for index, word in enumerate(itos)}
    model = SentimentLSTM(
        vocab_size=len(itos),
        embed_dim=checkpoint["args"]["embed_dim"],
        hidden_size=checkpoint["args"]["hidden_size"],
        num_layers=checkpoint["args"]["num_layers"],
        dropout=checkpoint["args"]["dropout"],
        pad_idx=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, stoi, itos, checkpoint["args"]["max_len"], checkpoint_path.stem


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    log_dir, figure_dir = resolve_path(args.log_dir), resolve_path(args.figure_dir)
    for d in (log_dir, figure_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 测试集只读一次，各模型复用同一批样本
    test_texts, test_labels = read_split(resolve_path(args.data_dir), "test")
    results = {}  # run_name -> {(mode, rate): acc}
    vocab_sizes = {}

    for ckpt_path in args.checkpoints:
        model, stoi, itos, max_len, run_name = load_model(resolve_path(ckpt_path), device)
        results[run_name] = {}
        vocab_sizes[run_name] = len(itos)
        base_dataset = ImdbDataset(test_texts, test_labels, stoi, max_len)
        print(f"\n模型 {run_name}（词表 {len(itos)}）:")
        for mode in args.modes:
            for rate in args.rates:
                # 每个 (mode, rate) 用独立且可复现的随机流
                offset = int(round(rate * 100)) * 2 + (0 if mode == "replace" else 1)
                rng = np.random.default_rng(args.seed + offset)
                samples = perturb_dataset(base_dataset, mode, rate, rng, len(itos))
                acc = accuracy_of(model, samples, device, args.batch_size)
                results[run_name][(mode, rate)] = acc
                print(f"  {mode:7s} p={rate:.1f}: acc {acc:.4f}")

    # CSV
    csv_path = log_dir / "noise_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "mode", "rate", "accuracy"])
        for run_name, by_cond in results.items():
            for (mode, rate), acc in sorted(by_cond.items()):
                writer.writerow([run_name, mode, f"{rate:.2f}", f"{acc:.4f}"])

    # 曲线：每种模式一个面板，每条权重一条线
    fig, axes = plt.subplots(1, len(args.modes), figsize=(6.5 * len(args.modes), 4.8))
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, args.modes):
        for run_name, by_cond in results.items():
            rates = sorted(r for m, r in by_cond if m == mode)
            accs = [by_cond[(mode, r)] for r in rates]
            ax.plot(rates, accs, marker="o", label=run_name)
        ax.set_xlabel("Noise rate p")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Noise type: {mode}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig_path = figure_dir / "noise_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"\n结果 CSV: {csv_path}")
    print(f"曲线图: {fig_path}")


if __name__ == "__main__":
    main()
