"""实验3：冻结 ResNet18 特征 + BERT-tiny 图文匹配训练入口。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import (
    CaptionDataset,
    collate_pairs,
    load_image_features,
    load_split,
    resolve_path,
)
from encoding import encode_text_dataset
from model import (
    ImageTextModel,
    count_parameters,
    infonce_loss,
    resolve_text_model_path,
    triplet_loss,
)
from retrieval import retrieval_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Flickr8k 图文检索训练")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--text-model", type=str, default="data/models/bert-tiny")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="infonce", choices=["infonce", "triplet"])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="投影头学习率")
    parser.add_argument("--text-lr", type=float, default=1e-4, help="BERT 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--freeze-text", action="store_true", help="冻结 BERT，只训练投影头")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--figure-dir", type=str, default="figures")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_with_features(data_dir: Path, features_dir: Path, split: str):
    image_ids, captions, _ = load_split(data_dir, split)
    feature_ids, features = load_image_features(features_dir, split)
    if feature_ids != image_ids:
        raise ValueError(
            f"{split} 的特征缓存和当前划分顺序不一致，请重新运行 precompute_features.py"
        )
    return image_ids, captions, features


@torch.no_grad()
def evaluate_split(
    model: ImageTextModel,
    tokenizer,
    features: torch.Tensor,
    captions: list[list[str]],
    args,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    image_embeddings = model.encode_image(features.to(device)).cpu()
    text_embeddings, image_indices = encode_text_dataset(
        model,
        tokenizer,
        captions,
        args.eval_batch_size,
        args.max_len,
        device,
        args.num_workers,
    )
    return retrieval_metrics(
        image_embeddings, text_embeddings, image_indices, num_images=len(captions)
    )


def train_one_epoch(
    model: ImageTextModel,
    tokenizer,
    dataset: CaptionDataset,
    features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    args,
    device: torch.device,
    epoch: int,
) -> float:
    dataset.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=partial(collate_pairs, tokenizer=tokenizer, max_length=args.max_len),
    )
    model.train()
    total_loss, steps = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
    for batch in progress:
        image_features = features[batch["image_index"]].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        image_embeddings = model.encode_image(image_features)
        text_embeddings = model.encode_text(input_ids, attention_mask)
        if args.loss == "infonce":
            loss = infonce_loss(image_embeddings, text_embeddings, args.temperature)
        else:
            loss = triplet_loss(image_embeddings, text_embeddings, args.margin)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        progress.set_postfix(loss=f"{total_loss / steps:.4f}")
    if steps == 0:
        raise RuntimeError("训练集为空或 batch_size 大于训练集")
    return total_loss / steps


def plot_curves(history: dict[str, list[float]], path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Train loss")
    axes[0].grid(alpha=0.3)

    for key, label in (
        ("dev_i2t_r1", "I2T R@1"),
        ("dev_i2t_r5", "I2T R@5"),
        ("dev_i2t_r10", "I2T R@10"),
    ):
        axes[1].plot(epochs, history[key], marker="o", markersize=3, label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].set_title(f"{title} - Dev retrieval")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_dir = resolve_path(args.data_dir)
    features_dir = resolve_path(args.features_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    log_dir = resolve_path(args.log_dir)
    figure_dir = resolve_path(args.figure_dir)
    for directory in (checkpoint_dir, log_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or (
        f"resnet18_bert-{args.loss}-d{args.embed_dim}"
        + ("-frozen" if args.freeze_text else "")
    )
    resolved_text_model, local_files_only = resolve_text_model_path(args.text_model)
    if not local_files_only:
        raise FileNotFoundError(
            f"找不到本地文本模型 {args.text_model}；离线环境请按 README 下载 bert-tiny"
        )

    train_ids, train_captions, train_features = load_split_with_features(
        data_dir, features_dir, "train"
    )
    dev_ids, dev_captions, dev_features = load_split_with_features(
        data_dir, features_dir, "dev"
    )
    print(f"实验名: {run_name}")
    print(f"设备: {device}")
    print(f"训练集: {len(train_ids)} 图 / {len(train_captions) * 5} 描述")
    print(f"验证集: {len(dev_ids)} 图 / {len(dev_captions) * 5} 描述")
    print(f"文本模型: {resolved_text_model}")
    print(f"损失: {args.loss} / 嵌入维度: {args.embed_dim}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_text_model, local_files_only=True)
    model = ImageTextModel(
        text_model_path=resolved_text_model,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        freeze_text=args.freeze_text,
    ).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    print(f"参数量: {total_parameters / 1e6:.3f}M，其中可训练 {trainable_parameters / 1e6:.3f}M")

    head_parameters = list(model.image_encoder.parameters()) + list(
        model.text_encoder.proj.parameters()
    )
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if not args.freeze_text:
        parameter_groups.append(
            {"params": model.text_encoder.bert.parameters(), "lr": args.text_lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    config = {
        **vars(args),
        "run_name": run_name,
        "device": str(device),
        "text_model_resolved": resolved_text_model,
        "train_images": len(train_ids),
        "dev_images": len(dev_ids),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    config_path = log_dir / f"{run_name}_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = log_dir / f"{run_name}.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "dev_i2t_r1",
        "dev_i2t_r5",
        "dev_i2t_r10",
        "dev_t2i_r1",
        "dev_t2i_r5",
        "dev_t2i_r10",
        "dev_i2t_median",
        "dev_t2i_median",
    ]
    history: dict[str, list[float]] = {key: [] for key in fieldnames[1:]}
    best_score = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    dataset = CaptionDataset(train_captions, one_per_image=True, seed=args.seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, tokenizer, dataset, train_features, optimizer, args, device, epoch
            )
            metrics = evaluate_split(
                model, tokenizer, dev_features, dev_captions, args, device
            )
            for key in history:
                value = train_loss if key == "train_loss" else metrics[key.removeprefix("dev_")]
                history[key].append(float(value))
            metric_values = [
                f"{metrics[key.removeprefix('dev_')]:.6f}" for key in fieldnames[2:]
            ]
            writer.writerow([epoch, f"{train_loss:.6f}", *metric_values])
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | "
                f"dev I2T R@1 {metrics['i2t_r1']:.2%} R@5 {metrics['i2t_r5']:.2%} "
                f"R@10 {metrics['i2t_r10']:.2%} | "
                f"T2I R@1 {metrics['t2i_r1']:.2%}"
            )

            score = (metrics["i2t_r1"], metrics["i2t_r5"], metrics["i2t_r10"])
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "metrics": metrics,
                        "text_model_resolved": resolved_text_model,
                    },
                    checkpoint_path,
                )
                print(f"  -> 保存最优模型（dev I2T R@1 {metrics['i2t_r1']:.2%}）")

    elapsed = time.time() - start
    time_path = log_dir / f"{run_name}_train_time.txt"
    time_path.write_text(f"{elapsed:.1f}\n", encoding="utf-8")
    plot_curves(history, figure_dir / f"{run_name}_curves.png", run_name)

    max_memory = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    print(
        f"\n训练结束，耗时 {elapsed / 60:.1f} 分钟，峰值显存 {max_memory:.2f} GB，"
        f"最优 epoch {best_epoch}，dev I2T R@1 {best_metrics.get('i2t_r1', 0):.2%}"
    )
    print(f"日志: {csv_path}")
    print(f"最优模型: {checkpoint_path}")
    print(f"训练耗时: {time_path}")


if __name__ == "__main__":
    main()
