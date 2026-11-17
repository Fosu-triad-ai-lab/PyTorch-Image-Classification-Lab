"""Flickr8k 数据读取、划分校验、图像预处理与特征缓存加载。"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
SPLITS = ("train", "dev", "test")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    return resolve_path(data_dir) if data_dir else DEFAULT_DATA_DIR


def resolve_image_dir(data_dir: Path) -> Path:
    # 原始 Flickr8k 压缩包里的目录名就是 Flicker8k_Dataset（少一个 r）
    for name in ("Flicker8k_Dataset", "Flickr8k_Dataset", "Images"):
        path = data_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"{data_dir} 下找不到图片目录，期望 Flicker8k_Dataset/ 或 Flickr8k_Dataset/"
    )


def resolve_text_dir(data_dir: Path) -> Path:
    path = data_dir / "Flickr8k_text"
    if not path.is_dir():
        raise FileNotFoundError(f"{data_dir} 下找不到 Flickr8k_text/")
    return path


def read_captions(token_path: Path) -> dict[str, list[str]]:
    """读取 Flickr8k.token.txt，返回 image_id -> 5 条描述。"""
    captions: dict[str, list[str]] = {}
    with token_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            key, caption = line.split("\t", 1)
            image_id = key.split("#", 1)[0].strip()
            # 原始文件里有一个坏名字 ...jpg.1，不是真实图片
            if not image_id.endswith(".jpg"):
                continue
            caption = " ".join(caption.split())
            if caption:
                captions.setdefault(image_id, []).append(caption)
    return captions


def read_split(text_dir: Path, split: str) -> list[str]:
    path = text_dir / f"Flickr_8k.{split}Images.txt"
    if not path.is_file():
        raise FileNotFoundError(f"找不到划分文件: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split(
    data_dir: str | Path | None = None, split: str = "train"
) -> tuple[list[str], list[list[str]], Path]:
    """返回 (image_ids, captions_per_image, image_dir)，只使用官方划分内的图片。"""
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split}")
    root = resolve_data_dir(data_dir)
    image_dir = resolve_image_dir(root)
    text_dir = resolve_text_dir(root)
    all_captions = read_captions(text_dir / "Flickr8k.token.txt")
    image_ids = read_split(text_dir, split)

    missing = [i for i in image_ids if i not in all_captions or not (image_dir / i).is_file()]
    if missing:
        raise FileNotFoundError(f"{split} 划分缺少 {len(missing)} 张图片或描述，例如 {missing[:3]}")

    captions = [all_captions[i] for i in image_ids]
    bad_counts = [i for i, caps in zip(image_ids, captions) if len(caps) != 5]
    if bad_counts:
        raise ValueError(f"{split} 中有 {len(bad_counts)} 张图片不是 5 条描述，例如 {bad_counts[:3]}")
    return image_ids, captions, image_dir


def build_image_transform() -> transforms.Compose:
    """冻结 ResNet18 使用的确定性预处理，不做随机增强。"""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


class ImagePathDataset(Dataset):
    """按 image_id 读取图片，用于预计算 CNN 特征。"""

    def __init__(self, image_dir: Path, image_ids: list[str], transform):
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        path = self.image_dir / self.image_ids[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)


class CaptionDataset(Dataset):
    """一条样本 = (图片在 split 中的下标, 一条描述)。

    one_per_image=True 时每个 epoch 为每张图随机抽一条描述，避免同一张图的
    多句描述同时进入 batch 被当成负样本；验证/测试时用全部 5 条描述。
    """

    def __init__(self, captions: list[list[str]], one_per_image: bool, seed: int = 42):
        self.captions = captions
        self.one_per_image = one_per_image
        self.seed = seed
        self.samples: list[tuple[int, str]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.one_per_image:
            rng = random.Random(self.seed + epoch)
            self.samples = [
                (image_index, rng.choice(caps))
                for image_index, caps in enumerate(self.captions)
                if caps
            ]
        else:
            self.samples = [
                (image_index, caption)
                for image_index, caps in enumerate(self.captions)
                for caption in caps
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_pairs(batch, tokenizer, max_length: int = 48):
    """把 (image_index, caption) 编码成 BERT 输入。"""
    image_indices = torch.tensor([item[0] for item in batch], dtype=torch.long)
    captions = [item[1] for item in batch]
    encoded = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "image_index": image_indices,
        **encoded,
        "captions": captions,
    }


def feature_path(features_dir: str | Path, split: str) -> Path:
    return resolve_path(features_dir) / f"resnet18_{split}.pt"


def load_image_features(
    features_dir: str | Path, split: str
) -> tuple[list[str], torch.Tensor]:
    path = feature_path(features_dir, split)
    if not path.is_file():
        raise FileNotFoundError(f"找不到特征缓存 {path}，请先运行 precompute_features.py")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    image_ids = list(obj["image_ids"])
    features = obj["features"].float()
    if features.shape[0] != len(image_ids):
        raise ValueError(f"{path} 中特征数和 image_ids 数不一致")
    return image_ids, features


def image_path(image_dir: Path, image_id: str) -> Path:
    return Path(image_dir) / image_id


def audit_dataset(data_dir: str | Path | None = None) -> None:
    """打印数据完整性摘要，供 README / 前置检查使用。"""
    root = resolve_data_dir(data_dir)
    image_dir = resolve_image_dir(root)
    text_dir = resolve_text_dir(root)
    all_captions = read_captions(text_dir / "Flickr8k.token.txt")
    all_images = {p.name for p in image_dir.glob("*.jpg")}
    split_images = {split: set(read_split(text_dir, split)) for split in SPLITS}
    used_images = set().union(*split_images.values())

    print(f"数据目录: {root}")
    print(f"图片目录: {image_dir}（{len(all_images)} 张 jpg）")
    print(f"描述文件: {text_dir / 'Flickr8k.token.txt'}（{len(all_captions)} 张图 / "
          f"{sum(len(v) for v in all_captions.values())} 条描述）")
    for split in SPLITS:
        ids = split_images[split]
        captions = sum(len(all_captions.get(i, [])) for i in ids)
        print(f"  {split}: {len(ids)} 张图 / {captions} 条描述")
    unused = set(all_captions) - used_images
    print(f"未参与划分的图片: {len(unused)}（{sum(len(all_captions[i]) for i in unused)} 条描述）")
    print("另有 1 个坏文件名 2258277193_586949ec62.jpg.1 在读 token 时跳过")
    print(f"磁盘上未被划分使用的真实图片: {len(all_images - used_images)} 张")


if __name__ == "__main__":
    audit_dataset()
