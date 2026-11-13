"""轻量卷积神经网络，BatchNorm 和 Dropout 可开关，用于对比实验。"""

import torch.nn as nn


def _make_dropout(use_dropout, dropout_p):
    return nn.Dropout(dropout_p) if use_dropout else nn.Identity()


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + ReLU，最后经最大池化输出。"""

    def __init__(self, in_channels, out_channels, use_bn):
        super().__init__()
        layers = []
        for input_channels, output_channels in (
            (in_channels, out_channels),
            (out_channels, out_channels),
        ):
            # 有 BatchNorm 时卷积不设偏置，减少冗余参数
            layers.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_bn,
                )
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.features(x))


class LightweightCNN(nn.Module):
    """VGG 风格的小型 CNN：3 个卷积块 + 1 个隐藏全连接层。

    输入 32x32，三个池化后为 4x4，特征维度 256*4*4。
    """

    def __init__(self, num_classes=10, use_bn=False, use_dropout=False, dropout_p=0.3):
        super().__init__()
        self.block1 = ConvBlock(3, 64, use_bn)
        self.block2 = ConvBlock(64, 128, use_bn)
        self.block3 = ConvBlock(128, 256, use_bn)
        self.stage_dropout = _make_dropout(use_dropout, dropout_p)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            _make_dropout(use_dropout, dropout_p),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            _make_dropout(use_dropout, dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stage_dropout(self.block1(x))
        x = self.stage_dropout(self.block2(x))
        x = self.stage_dropout(self.block3(x))
        return self.classifier(x)
