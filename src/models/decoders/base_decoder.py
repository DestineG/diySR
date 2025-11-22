# src/models/decoders/base_decoder.py

import torch.nn as nn

from . import register_decoder

base_decoder_defaultConfig = {
    'in_channels': 64,
    'out_channels': 3,
}
@register_decoder('base_decoder')
class BaseDecoder(nn.Module):
    def __init__(self, decoder_config=base_decoder_defaultConfig):
        super(BaseDecoder, self).__init__()
        self.decoder = nn.Conv2d(decoder_config['in_channels'], decoder_config['out_channels'], kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.decoder(x)
        x = self.relu(x)
        return x