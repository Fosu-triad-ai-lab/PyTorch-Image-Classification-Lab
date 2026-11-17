"""用冻结的 torchvision ResNet18 预计算 Flickr8k 图像特征。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from data import (
    BASE_DIR,
    ImagePathDataset,
    build_image_transform,
    feature_path,
    load_split,
    resolve_data_dir,
)

FEATURE_DIM = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Flickr8k ResNet18 图像特征预计算")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/features")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model = model.eval().to(device)
    transform = build_image_transform()
    print(f"设备: {device}")
    print(f"backbone: resnet18 / {weights} / 输出 {FEATURE_DIM} 维，冻结不训练")

    for split in args.splits:
        image_ids, _, image_dir = load_split(data_dir, split)
        dataset = ImagePathDataset(image_dir, image_ids, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        features, start = [], time.time()
        with torch.no_grad():
            for images in loader:
                features.append(model(images.to(device, non_blocking=True)).cpu())
        features = torch.cat(features, dim=0)
        if features.shape != (len(image_ids), FEATURE_DIM):
            raise RuntimeError(f"{split} 特征形状异常: {tuple(features.shape)}")

        path = feature_path(output_dir, split)
        torch.save(
            {
                "image_ids": image_ids,
                "features": features,
                "backbone": "resnet18",
                "weights": str(weights),
                "feature_dim": FEATURE_DIM,
                "transform": "resize256_centercrop224_imagenet_norm",
            },
            path,
        )
        print(
            f"{split}: {len(image_ids)} 张图，特征 {tuple(features.shape)}，"
            f"耗时 {time.time() - start:.1f} 秒 -> {path}"
        )


if __name__ == "__main__":
    main()
