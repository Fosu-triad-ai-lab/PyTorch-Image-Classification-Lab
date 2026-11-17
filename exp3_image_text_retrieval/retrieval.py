"""图像-文本检索指标：Recall@K 与中位排名。"""

from __future__ import annotations

import torch


def retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_indices: torch.Tensor,
    num_images: int,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """计算图像检索文本和文本检索图像两个方向的指标。

    image_indices[j] 是第 j 条描述所属图片在 split 中的下标。图像检索文本时，
    一张图有 5 条正确描述，只要任意一条进入 top-K 就算命中。
    """
    image_embeddings = image_embeddings.detach().cpu()
    text_embeddings = text_embeddings.detach().cpu()
    image_indices = image_indices.detach().cpu().long()
    similarity = image_embeddings @ text_embeddings.t()

    # 图像 -> 文本：每条描述在每张图下的排名
    order_i2t = similarity.argsort(dim=1, descending=True)
    rank_i2t = order_i2t.argsort(dim=1) + 1
    gt_mask = image_indices.unsqueeze(0) == torch.arange(num_images).unsqueeze(1)
    best_gt_rank = rank_i2t.masked_fill(~gt_mask, 10**9).min(dim=1).values

    # 文本 -> 图像：每个文本 query 下正确图片的排名
    order_t2i = similarity.t().argsort(dim=1, descending=True)
    rank_t2i = order_t2i.argsort(dim=1) + 1
    gt_image_rank = rank_t2i[torch.arange(len(image_indices)), image_indices]

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = float((best_gt_rank <= k).float().mean())
        metrics[f"t2i_r{k}"] = float((gt_image_rank <= k).float().mean())
    metrics["i2t_median"] = float(best_gt_rank.float().median())
    metrics["t2i_median"] = float(gt_image_rank.float().median())
    return metrics
