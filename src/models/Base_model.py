# src/models/Base_model.py

import torch.nn as nn
import torch.nn.functional as F

from . import register_model, Model
from src.models.decoders import get_decoder_by_name
from src.models.encoders import get_encoder_by_name

base_model_defaultConfig = {
    'phase': 'train',
    'encoder': {
        'encoderClsName': 'base_encoder',
        'encoderArgs': {
            'in_channels': 3,
            'out_channels': 64,
        }
    },
    'decoder': {
        'decoderClsName': 'base_decoder',
        'decoderArgs': {
            'in_channels': 64,
            'out_channels': 3,
        }
    }
}
@register_model('base_model')
class BaseModel(Model):
    def __init__(self, model_config=base_model_defaultConfig):
        super(BaseModel, self).__init__()
        self.phase = model_config.get('phase', 'train')

        self.encoder_config = model_config.get('encoder', {})
        self.encoder_name = self.encoder_config.get('encoderClsName', 'base_encoder')
        self.encoder_args = self.encoder_config.get('encoderArgs', {})
        self.encoder = get_encoder_by_name(self.encoder_name)(encoder_config=self.encoder_args)

        self.decoder_config = model_config.get('decoder', {})
        self.decoder_name = self.decoder_config.get('decoderClsName', 'base_decoder')
        self.decoder_args = self.decoder_config.get('decoderArgs', {})
        self.decoder = get_decoder_by_name(self.decoder_name)(decoder_config=self.decoder_args)

    def forward(self, x):

        lr = x['lr']

        # ----------- TRAIN: use HR shape -----------
        if self.phase == 'train':
            hr_shape = x['hr'].shape      # B,C,H,W
            out_base = F.interpolate(
                lr,
                size=hr_shape[2:],        # (H, W)
                mode='bicubic',
                align_corners=False
            )

            encoded = self.encoder(lr)
            decoded = self.decoder(encoded)

        # ----------- TEST INFERENCE: use scale factor ---------------
        else:
            scale = x['scale']
            out_base = F.interpolate(
                lr,
                scale_factor=scale,
                mode='bicubic',
                align_corners=False
            )

            encoded = self.encoder(lr)
            decoded = self.decoder(encoded)

        return {
            'encoded': encoded,
            'decoded': decoded,
            'out_base': out_base
        }