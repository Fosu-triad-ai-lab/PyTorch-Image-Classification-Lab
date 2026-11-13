"""CIFAR-10 数据集加载、划分与预处理。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class TransformSubset(Dataset):
    """按索引访问原始数据集，并对样本应用指定预处理。"""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms():
    """训练集做随机裁剪和水平翻转，验证/测试集只做张量化和归一化。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(data_dir, batch_size, num_workers, val_ratio=0.1, seed=42):
    """返回 (train_loader, val_loader, test_loader)。

    官方训练集 50000 张按 9:1 划成训练集 45000 张、验证集 5000 张，
    官方测试集 10000 张只用于最终评估。
    """
    data_dir = Path(data_dir).expanduser()
    train_transform, eval_transform = build_transforms()

    train_base = CIFAR10(root=str(data_dir), train=True, download=True, transform=None)
    test_base = CIFAR10(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )

    val_size = int(len(train_base) * val_ratio)
    train_size = len(train_base) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(
        train_base, [train_size, val_size], generator=generator
    )

    train_set = TransformSubset(train_base, train_split.indices, train_transform)
    val_set = TransformSubset(train_base, val_split.indices, eval_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_base, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
