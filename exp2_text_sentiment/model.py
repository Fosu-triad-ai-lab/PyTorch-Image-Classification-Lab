"""Embedding + LSTM 情感二分类模型，支持随机初始化或 GloVe 预训练词向量。"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """词向量 -> Dropout -> LSTM（取最后时刻隐状态）-> Dropout -> 全连接二分类。"""

    def __init__(
        self,
        vocab_size,
        embed_dim=100,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
        pad_idx=0,
        embedding_matrix=None,
    ):
        super().__init__()
        if embedding_matrix is not None:
            # 预训练词向量初始化，训练中继续微调（freeze=False）
            self.embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(embedding_matrix), freeze=False, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n 形状 (num_layers, batch, hidden)，取最后一层最后时刻的隐状态
        return self.classifier(self.dropout(h_n[-1]))
