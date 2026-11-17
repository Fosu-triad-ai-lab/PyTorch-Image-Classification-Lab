"""图像-文本双塔：冻结 ResNet18 特征 + 预训练 BERT-tiny 文本编码器。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FEATURE_DIM = 512


def resolve_text_model_path(model_path: str | Path) -> tuple[str, bool]:
    path = Path(model_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(BASE_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve()), True
    # 允许直接传 HuggingFace 模型名；离线环境建议传本地目录
    return str(model_path), False


class ProjectionHead(nn.Module):
    """把单模态特征投到共享嵌入空间。"""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ImageEncoder(nn.Module):
    def __init__(self, input_dim: int = IMAGE_FEATURE_DIM, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = ProjectionHead(input_dim, embed_dim, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(features), dim=-1)


class TextEncoder(nn.Module):
    """BERT-tiny + mean pooling + 投影；默认微调，也可冻结。"""

    def __init__(
        self,
        model_path: str | Path,
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        super().__init__()
        resolved_path, local_files_only = resolve_text_model_path(model_path)
        self.bert = AutoModel.from_pretrained(resolved_path, local_files_only=local_files_only)
        self.proj = ProjectionHead(self.bert.config.hidden_size, embed_dim, dropout)
        self.freeze = freeze
        if freeze:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.bert.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(self.proj(pooled), dim=-1)


class ImageTextModel(nn.Module):
    def __init__(
        self,
        text_model_path: str | Path,
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze_text: bool = False,
        image_input_dim: int = IMAGE_FEATURE_DIM,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(image_input_dim, embed_dim, dropout)
        self.text_encoder = TextEncoder(text_model_path, embed_dim, dropout, freeze_text)

    def encode_image(self, features: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(features)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)


def infonce_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """对称 InfoNCE：batch 内其他样本作为负样本。"""
    logits = image_embeddings @ text_embeddings.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_image_to_text = F.cross_entropy(logits, labels)
    loss_text_to_image = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_image_to_text + loss_text_to_image)


def triplet_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """batch-hard 三元组：每个 anchor 取最难负样本。"""
    similarity = image_embeddings @ text_embeddings.t()
    positive = similarity.diag()
    mask = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    hardest_text = similarity.masked_fill(mask, -1.0).max(dim=1).values
    hardest_image = similarity.t().masked_fill(mask, -1.0).max(dim=1).values
    loss_image = F.relu(margin - positive + hardest_text).mean()
    loss_text = F.relu(margin - positive + hardest_image).mean()
    return 0.5 * (loss_image + loss_text)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
