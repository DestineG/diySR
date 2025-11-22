# src/models/encoders/base_encoder.py

import torch.nn as nn

from . import register_encoder

base_encoder_defaultConfig = {
    'in_channels': 3,
    'out_channels': 64,
}
@register_encoder('base_encoder')
class BaseEncoder(nn.Module):
    def __init__(self, encoder_config=base_encoder_defaultConfig):
        super(BaseEncoder, self).__init__()
        self.encoder = nn.Conv2d(encoder_config['in_channels'], encoder_config['out_channels'], kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        return x