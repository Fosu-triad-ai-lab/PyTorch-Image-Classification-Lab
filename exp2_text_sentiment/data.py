"""aclImdb 数据加载、清洗分词、词表构建与 GloVe 词向量加载。"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
LABEL_NAMES = ("negative", "positive")

_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    """清洗（去 HTML 标签）+ 小写化 + 按字母数字切词。"""
    text = _TAG_RE.sub(" ", text)
    return _TOKEN_RE.findall(text.lower())


def read_split(data_dir, split):
    """读取 aclImdb 某个划分的全部评论，返回 (texts, labels)，neg=0，pos=1。"""
    data_dir = Path(data_dir)
    texts, labels = [], []
    for label, name in ((0, "neg"), (1, "pos")):
        files = sorted((data_dir / split / name).glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"未找到 {data_dir / split / name} 下的评论文本")
        for path in files:
            texts.append(path.read_text(encoding="utf-8"))
            labels.append(label)
    return texts, labels


def build_vocab(texts, vocab_size, min_freq=2):
    """按词频从高到低构建词表，前两位固定留给 <pad> 和 <unk>。"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    words = [w for w, count in counter.most_common() if count >= min_freq]
    words = words[: vocab_size - len(SPECIAL_TOKENS)]
    itos = list(SPECIAL_TOKENS.keys()) + words
    stoi = {word: index for index, word in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, max_len):
    """分词并转成 id 序列，超长截断；保证至少 1 个 token（空序列用 <unk>）。"""
    unk = stoi["<unk>"]
    ids = [stoi.get(word, unk) for word in tokenize(text)[:max_len]]
    if not ids:
        ids = [unk]
    return torch.tensor(ids, dtype=torch.long)


class ImdbDataset(Dataset):
    """预先编码好的 IMDB 数据集，样本是 (id 序列, 标签)。"""

    def __init__(self, texts, labels, stoi, max_len):
        self.samples = [
            (encode(text, stoi, max_len), torch.tensor(label, dtype=torch.long))
            for text, label in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):
    """变长序列按 batch 内最长补齐，同时返回真实长度供 pack_padded 使用。"""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)


def split_train_val(texts, labels, val_ratio=0.1):
    """用固定随机数流把训练集按比例切成训练/验证两份。"""
    order = np.random.permutation(len(texts))
    val_size = int(len(texts) * val_ratio)
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    pick = lambda idx: ([texts[i] for i in idx], [labels[i] for i in idx])
    return (*pick(train_idx), *pick(val_idx))


def build_dataloaders(data_dir, stoi, batch_size, max_len, val_ratio=0.1, num_workers=0):
    """返回 (train_loader, val_loader, test_loader, train_texts, train_labels)。"""
    train_texts, train_labels = read_split(data_dir, "train")
    train_texts, train_labels, val_texts, val_labels = split_train_val(
        train_texts, train_labels, val_ratio
    )
    test_texts, test_labels = read_split(data_dir, "test")

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(ImdbDataset(train_texts, train_labels, stoi, max_len), **loader_kwargs)
    val_loader = DataLoader(
        ImdbDataset(val_texts, val_labels, stoi, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ImdbDataset(test_texts, test_labels, stoi, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, (train_texts, train_labels)


def load_glove_matrix(itos, glove_path):
    """从 GloVe 文本文件加载词向量。

    返回 (embedding_matrix, hit_rate)：命中的词用 GloVe 向量，
    未命中的词和 <unk> 用已加载向量的均值，<pad> 全零。
    """
    needed = {w for w in itos if w not in SPECIAL_TOKENS}
    vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            if word in needed:
                vectors[word] = np.array(rest.split(), dtype=np.float32)
    if not vectors:
        raise ValueError(f"{glove_path} 中没有任何词命中词表，请检查文件")
    embed_dim = len(next(iter(vectors.values())))

    matrix = np.empty((len(itos), embed_dim), dtype=np.float32)
    mean_vec = np.mean(np.stack(list(vectors.values())), axis=0)
    for index, word in enumerate(itos):
        if word == "<pad>":
            matrix[index] = 0.0
        elif word in vectors:
            matrix[index] = vectors[word]
        else:
            # 未命中词和 <unk> 都用全体 GloVe 向量的均值做初始值
            matrix[index] = mean_vec
    hit_rate = len(vectors) / (len(itos) - len(SPECIAL_TOKENS))
    return matrix, hit_rate
