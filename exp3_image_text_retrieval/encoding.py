"""批量编码描述文本，训练和评估共用。"""

from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader

from data import CaptionDataset, collate_pairs


@torch.no_grad()
def encode_text_dataset(
    model,
    tokenizer,
    captions: list[list[str]],
    batch_size: int,
    max_len: int,
    device: torch.device,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = CaptionDataset(captions, one_per_image=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_pairs, tokenizer=tokenizer, max_length=max_len),
    )
    embeddings, image_indices = [], []
    for batch in loader:
        text_embeddings = model.encode_text(
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True),
        )
        embeddings.append(text_embeddings.cpu())
        image_indices.append(batch["image_index"])
    return torch.cat(embeddings), torch.cat(image_indices)
