# src/models/decoders/base_decoder.py

import torch.nn as nn
from . import register_decoder
from ..blocks import ResidualConv


base_decoder_defaultConfig = {
    'in_channels': 64,
    'out_channels': 3,
}
@register_decoder('base_decoder')
class BaseDecoder(nn.Module):
    def __init__(self, decoder_config=base_decoder_defaultConfig):
        super(BaseDecoder, self).__init__()

        # 整个解码器模块
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_config['in_channels'], 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            ResidualConv(32),
            nn.Conv2d(32, decoder_config['out_channels'], kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)
