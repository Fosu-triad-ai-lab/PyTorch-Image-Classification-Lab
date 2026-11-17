"""实验3：测试集图文检索评估与定性示例。"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoTokenizer

from data import image_path, load_image_features, load_split, resolve_path
from encoding import encode_text_dataset
from model import ImageTextModel, resolve_text_model_path
from retrieval import retrieval_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Flickr8k 图文检索评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument("--text-model", type=str, default="", help="覆盖 checkpoint 中的文本模型路径")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--figure-dir", type=str, default="figures")
    return parser.parse_args()


def get_caption(captions: list[list[str]], image_indices: torch.Tensor, text_index: int) -> str:
    """根据描述在展平列表中的下标找回原文。"""
    image_index = int(image_indices[text_index])
    offset = int((image_indices[:text_index] == image_index).sum())
    return captions[image_index][offset]


def plot_examples(
    image_dir: Path,
    image_ids: list[str],
    captions: list[list[str]],
    similarity: torch.Tensor,
    image_indices: torch.Tensor,
    example_indices: list[int],
    path: Path,
    title: str,
) -> None:
    if not example_indices:
        return
    columns = 4
    rows = math.ceil(len(example_indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.4))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, index in zip(axes, example_indices):
        top1 = int(similarity[index].argmax())
        pred = get_caption(captions, image_indices, top1)
        with Image.open(image_path(image_dir, image_ids[index])) as image:
            ax.imshow(image.convert("RGB"))
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"pred: {pred[:48]}\nGT: {captions[index][0][:48]}\n"
            f"score {similarity[index, top1]:.3f}",
            fontsize=7,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_args = checkpoint["args"]
    run_name = checkpoint_path.stem

    text_model = args.text_model or checkpoint.get("text_model_resolved") or train_args.get(
        "text_model", "data/models/bert-tiny"
    )
    resolved_text_model, local_files_only = resolve_text_model_path(text_model)
    if not local_files_only:
        raise FileNotFoundError(
            f"找不到本地文本模型 {text_model}；可用 --text-model 指向 bert-tiny 目录"
        )

    data_dir = resolve_path(args.data_dir)
    features_dir = resolve_path(args.features_dir)
    log_dir = resolve_path(args.log_dir)
    figure_dir = resolve_path(args.figure_dir)
    for directory in (log_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_ids, captions, image_dir = load_split(data_dir, args.split)
    feature_ids, features = load_image_features(features_dir, args.split)
    if feature_ids != image_ids:
        raise ValueError("特征缓存和当前划分顺序不一致，请重新运行 precompute_features.py")

    tokenizer = AutoTokenizer.from_pretrained(resolved_text_model, local_files_only=True)
    model = ImageTextModel(
        text_model_path=resolved_text_model,
        embed_dim=train_args["embed_dim"],
        dropout=train_args["dropout"],
        freeze_text=train_args.get("freeze_text", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        image_embeddings = model.encode_image(features.to(device)).cpu()
        text_embeddings, image_indices = encode_text_dataset(
            model,
            tokenizer,
            captions,
            args.batch_size,
            train_args["max_len"],
            device,
        )

    metrics = retrieval_metrics(
        image_embeddings, text_embeddings, image_indices, num_images=len(image_ids)
    )
    similarity = image_embeddings @ text_embeddings.t()

    suffix = "test" if args.split == "test" else args.split
    report_path = log_dir / f"{run_name}_{suffix}.txt"
    report_lines = [
        f"模型: {checkpoint_path}",
        f"训练轮次: {checkpoint.get('epoch')}",
        f"评估划分: {args.split}",
        f"样本: {len(image_ids)} 图 / {text_embeddings.size(0)} 条描述",
        f"图像 -> 文本 Recall@1: {metrics['i2t_r1']:.4f}",
        f"图像 -> 文本 Recall@5: {metrics['i2t_r5']:.4f}",
        f"图像 -> 文本 Recall@10: {metrics['i2t_r10']:.4f}",
        f"图像 -> 文本 中位排名: {metrics['i2t_median']:.0f}",
        f"文本 -> 图像 Recall@1: {metrics['t2i_r1']:.4f}",
        f"文本 -> 图像 Recall@5: {metrics['t2i_r5']:.4f}",
        f"文本 -> 图像 Recall@10: {metrics['t2i_r10']:.4f}",
        f"文本 -> 图像 中位排名: {metrics['t2i_median']:.0f}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    count = min(args.max_examples, len(image_ids))
    example_indices = random.Random(args.example_seed).sample(range(len(image_ids)), count)
    examples_path = log_dir / f"{run_name}_{suffix}_examples.csv"
    with examples_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "top1_caption", "top1_score", "ground_truth_captions"]
        )
        for index in example_indices:
            top1 = int(similarity[index].argmax())
            writer.writerow(
                [
                    image_ids[index],
                    get_caption(captions, image_indices, top1),
                    f"{similarity[index, top1]:.4f}",
                    " | ".join(captions[index]),
                ]
            )

    plot_examples(
        image_dir,
        image_ids,
        captions,
        similarity,
        image_indices,
        example_indices,
        figure_dir / f"{run_name}_{suffix}_examples.png",
        run_name,
    )

    print(f"评估划分: {args.split}")
    print(
        f"图像->文本 R@1 {metrics['i2t_r1']:.2%} R@5 {metrics['i2t_r5']:.2%} "
        f"R@10 {metrics['i2t_r10']:.2%}，中位排名 {metrics['i2t_median']:.0f}"
    )
    print(
        f"文本->图像 R@1 {metrics['t2i_r1']:.2%} R@5 {metrics['t2i_r5']:.2%} "
        f"R@10 {metrics['t2i_r10']:.2%}，中位排名 {metrics['t2i_median']:.0f}"
    )
    print(f"评估报告: {report_path}")
    print(f"示例 CSV: {examples_path}")


if __name__ == "__main__":
    main()
