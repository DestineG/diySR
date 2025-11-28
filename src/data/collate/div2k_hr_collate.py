# /src/data/collate/div2k_hr_collate.py

import torch
import torch.nn.functional as F
import random
from . import register_collate


@register_collate('div2k_hr')
def div2k_hr_collate(batch, config=None):
    mode = config.get('mode')
    interp_mode = config.get('interp_mode')
    normalize_config = config.get('normalize_config')
    normalize = normalize_config.get('normalize')
    normalize_mean = normalize_config.get('mean')
    normalize_std = normalize_config.get('std')

    if mode in ['train', 'val']:
        min_scale = config.get('min_scale')
        max_scale = config.get('max_scale')
        patch_size = config.get('patch_size')
        # 随机 crop
        hr_patches = []
        for img in batch:
            _, H, W = img.shape
            top = random.randint(0, H - patch_size)
            left = random.randint(0, W - patch_size)
            crop = img[:, top:top + patch_size, left:left + patch_size]
            hr_patches.append(crop)
        hr_tensor = torch.stack(hr_patches, dim=0)
        # 随机 scale
        scale = random.randint(min_scale, max_scale)
        lr_tensor = F.interpolate(
            hr_tensor, scale_factor=1/scale,
            mode=interp_mode, align_corners=False
        )
        # 归一化
        if normalize:
            mean = torch.tensor(normalize_mean, device=lr_tensor.device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=lr_tensor.device).view(1, -1, 1, 1)
            hr_tensor = (hr_tensor / 255.0 - mean) / std
            lr_tensor = (lr_tensor / 255.0 - mean) / std

        return {"lr": lr_tensor, "scale":  scale}, hr_tensor
    
    elif mode == 'test':
        scale = config.get('scale')
        hr_list = []
        lr_list = []
        max_hw_sum = 2040 + 1356  # H+W 最大和

        for hr in batch:  # batch 是 list，每个元素 [C,H,W]
            C, H, W = hr.shape

            # 判断 H+W 是否超过阈值
            if H + W > max_hw_sum:
                if W > H:
                    crop_h, crop_w = 1356, 2040
                else:
                    crop_h, crop_w = 2040, 1356

                # 中心裁剪
                start_h = (H - crop_h) // 2
                start_w = (W - crop_w) // 2
                hr = hr[:, start_h:start_h+crop_h, start_w:start_w+crop_w]

            # 下采样生成 LR
            lr_h = hr.shape[-2] // scale
            lr_w = hr.shape[-1] // scale
            lr = F.interpolate(hr.unsqueeze(0), size=(lr_h, lr_w), mode=interp_mode, align_corners=False).squeeze(0)

            hr_list.append(hr)
            lr_list.append(lr)

        hr_tensor = torch.stack(hr_list, dim=0)
        lr_tensor = torch.stack(lr_list, dim=0)

        if normalize:
            mean = torch.tensor(normalize_mean, device=lr_tensor.device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=lr_tensor.device).view(1, -1, 1, 1)
            hr_tensor = (hr_tensor / 255.0 - mean) / std
            lr_tensor = (lr_tensor / 255.0 - mean) / std

        return {"lr": lr_tensor, "scale":  scale}, hr_tensor

    else:
        raise ValueError(f"Unsupported mode: {mode}")