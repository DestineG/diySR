# src/models/blocks.py

import torch.nn as nn


class ResidualConv(nn.Module):
    """简单残差卷积块，不改变通道和尺寸"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(x + self.block(x))