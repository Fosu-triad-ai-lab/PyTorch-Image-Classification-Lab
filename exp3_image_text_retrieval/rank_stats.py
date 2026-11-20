"""实验3：真值排名分布与失败样本统计（报告 8.1 的数字来源）。

对指定权重做一次完整检索评估，统计每张图"5 条真值描述中最好排名"的分布，
以及对 top1 失败样本的进一步分析：真值是否仍在 top10、完全错配比例、
top1 与最佳真值的相似度差距。结果写入 logs/<run>_rank_stats.txt。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data import load_image_features, load_split, resolve_path
from encoding import encode_text_dataset
from model import ImageTextModel, resolve_text_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Flickr8k 检索真值排名分布统计")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-model", type=str, default="", help="覆盖 checkpoint 中的文本模型路径")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


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
        raise FileNotFoundError(f"找不到本地文本模型 {text_model}")

    data_dir = resolve_path(args.data_dir)
    features_dir = resolve_path(args.features_dir)
    log_dir = resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    image_ids, captions, _ = load_split(data_dir, args.split)
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

    similarity = image_embeddings @ text_embeddings.t()
    num_images = len(image_ids)

    # 每张图 5 条真值描述在全部候选中的最好排名（与 retrieval.py 同口径）
    order = similarity.argsort(dim=1, descending=True)
    rank = order.argsort(dim=1) + 1
    gt_mask = image_indices.unsqueeze(0) == torch.arange(num_images).unsqueeze(1)
    best_gt_rank = rank.masked_fill(~gt_mask, 10**9).min(dim=1).values
    top1_sim = similarity.max(dim=1).values
    # 每张图最佳真值描述的相似度
    gt_sim = similarity.masked_fill(~gt_mask, float("-inf")).max(dim=1).values

    buckets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 200), (201, 5000)]
    lines = [
        f"模型: {checkpoint_path}",
        f"评估划分: {args.split}（{num_images} 图 / {text_embeddings.size(0)} 条描述）",
        "",
        "真值最佳排名分布（每张图 5 条真值描述中最好的排名）:",
    ]
    counts = []
    for low, high in buckets:
        count = int(((best_gt_rank >= low) & (best_gt_rank <= high)).sum())
        counts.append(count)
        lines.append(f"  排名 {low:>3}-{high:<4}: {count:>4} 张（{count / num_images:.1%}）")
    lines.append(f"  合计: {sum(counts)} 张")

    failed = best_gt_rank > 1
    fail_count = int(failed.sum())
    in_top10 = int((best_gt_rank[failed] <= 10).sum())
    beyond_100 = int((best_gt_rank[failed] > 100).sum())
    sim_gap = (top1_sim - gt_sim)[failed]
    lines += [
        "",
        f"top1 失败样本: {fail_count} 张",
        f"  失败中真值仍在 top10 内: {in_top10} 张（{in_top10 / fail_count:.1%}）",
        f"  失败中排名 100 开外（完全错配）: {beyond_100} 张（{beyond_100 / fail_count:.1%}）",
        f"  top1 与最佳真值相似度差: 中位数 {sim_gap.median():.3f}，"
        f"小于 0.05 的占 {(sim_gap < 0.05).float().mean():.1%}",
    ]

    report_path = log_dir / f"{run_name}_rank_stats.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n统计报告: {report_path}")


if __name__ == "__main__":
    main()
