# src/models/Base_model.py

import sys
import torch.nn as nn
import torch.nn.functional as F

from . import register_model, Model
from src.models.decoders import get_decoder_by_name
from src.models.encoders import get_encoder_by_name

# k 为奇数；dilation 为采样间隔
def extract_localPatch(encoded, k=3, dilation=1):
    padding = (k // 2) * dilation

    # unfold -> [B, C*k*k, H*W]
    patches = F.unfold(encoded, kernel_size=k, padding=padding, dilation=dilation)

    # reshape to [B, C, k*k, H, W]
    B, CK2, HW = patches.shape
    C = encoded.shape[1]
    H, W = encoded.shape[2], encoded.shape[3]

    patches = patches.view(B, C, k*k, H, W)
    return patches

base_model_defaultConfig = {
    'phase': 'train',
    'local_patch_k': 3,
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
            'in_channels': 64*9,   # local_patch_k^2
            'out_channels': 3,
        }
    }
}
@register_model('base_model')
class BaseModel(Model):
    def __init__(self, model_config=base_model_defaultConfig):
        super(BaseModel, self).__init__()
        self.config = model_config
        self.phase = self.config.get('phase', 'train')

        self.encoder_config = self.config.get('encoder', {})
        self.encoder_name = self.encoder_config.get('encoderClsName', 'base_encoder')
        self.encoder_args = self.encoder_config.get('encoderArgs', {})
        self.encoder = get_encoder_by_name(self.encoder_name)(encoder_config=self.encoder_args)

        self.decoder_config = self.config.get('decoder', {})
        self.decoder_name = self.decoder_config.get('decoderClsName', 'base_decoder')
        self.decoder_args = self.decoder_config.get('decoderArgs', {})
        self.decoder = get_decoder_by_name(self.decoder_name)(decoder_config=self.decoder_args)

    def forward(self, x):

        lr = x['lr']

        # ----------- TRAIN: use HR shape -----------
        if self.phase == 'train':
            hr_shape = x['hr'].shape      # B,C,H,W
            # out_base = F.interpolate(
            #     lr,
            #     size=hr_shape[2:],        # (H, W)
            #     mode='bicubic',
            #     align_corners=False
            # )
            target_h, target_w = hr_shape[2], hr_shape[3]

        # ----------- TEST INFERENCE: use scale factor ---------------
        else:
            scale = x['scale']
            # out_base = F.interpolate(
            #     lr,
            #     scale_factor=scale,
            #     mode='bicubic',
            #     align_corners=False
            # )
            target_h, target_w = scale * lr.shape[2], scale * lr.shape[3]

        # ----------- Encoder + Local Feature -----------
        encoded = self.encoder(lr)
        local_feature = extract_localPatch(encoded, k=3, dilation=1)
        lf_B, lf_C, lf_K2, lf_H, lf_W = local_feature.shape
        local_feature = local_feature.view(lf_B, lf_C*lf_K2, lf_H, lf_W)

        # print("Encoded shape:", encoded.shape)
        # print("Local feature shape before resize:", local_feature.shape)

        # ----------- 插值到输出图像尺寸(假设特征曲面的连续性) -----------
        local_feature = F.interpolate(
            local_feature,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        # print("Local feature shape after resize:", local_feature.shape)

        # ----------- Decoder -----------
        decoded = self.decoder(local_feature)

        return {
            'decoded': decoded
        }