# src/models/Base_model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import get_decoder_by_name
from .encoders import get_encoder_by_name


registered_models = {}
def register_model(name):
    def decorator(cls):
        registered_models[name] = cls
        return cls
    return decorator

# 根据名称获取已注册的model
def get_model_by_name(name):
    """
    根据名称返回已注册的model类
    """
    model = registered_models.get(name)
    if model is None:
        raise ValueError(f"Model with name '{name}' is not registered.")
    return model

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

@register_model('base_model')
class BaseModel(Model):
    def __init__(self, config=None):
        super(BaseModel, self).__init__()
        self.config = config
        self.local_patch_k = self.config.get('local_patch_k', 3)

        self.encoder_config = self.config.get('encoder_config', {})
        self.encoder_name = self.encoder_config.get('encoderClsName', 'base_encoder')
        self.encoderClsArgs = self.encoder_config.get('encoderClsArgs', {})
        self.encoder = get_encoder_by_name(self.encoder_name)(encoder_config=self.encoderClsArgs)

        self.decoder_config = self.config.get('decoder_config', {})
        self.decoder_name = self.decoder_config.get('decoderClsName', 'base_decoder')
        self.decoderClsArgs = self.decoder_config.get('decoderClsArgs', {})
        self.decoder = get_decoder_by_name(self.decoder_name)(decoder_config=self.decoderClsArgs)

    # input: {"lr": , "scale": }
    def forward(self, input):
        x = input["lr"]
        scale = input["scale"]
        target_h, target_w = scale * x.shape[2], scale * x.shape[3]

        # ----------- Encoder + Local Feature -----------
        encoded = self.encoder(x)
        local_feature = extract_localPatch(encoded, k=self.local_patch_k, dilation=1)
        lf_B, lf_C, lf_K2, lf_H, lf_W = local_feature.shape
        local_feature = local_feature.view(lf_B, lf_C*lf_K2, lf_H, lf_W)

        # ----------- 插值到输出图像尺寸(假设特征曲面的连续性) -----------
        local_feature = F.interpolate(
            local_feature,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )

        # ----------- Decoder -----------
        decoded = self.decoder(local_feature)

        return {
            "feature": {
                "encoded": encoded
            },
            "pred": decoded
        }