# /src/data/collate/div2k_hr_collate.py

import torch
import torch.nn.functional as F
import random
from . import register_collate

div2k_hr_defaultConfig = {
    'phase': 'train',       # train / val / test / infer
    'patch_size': 128,       # HR random crop for training
    'min_scale': 1,
    'max_scale': 4,
    'scale': 2,
    'interp_mode': 'bicubic'
}
@register_collate('div2k_hr')
def div2k_hr_collate(batch, collate_config=div2k_hr_defaultConfig):
    phase = collate_config.get('phase', 'train')

    # ------------------------------
    # 1. train/val: batch = HR images
    # ------------------------------
    if phase in ['train', 'val']:
        patch = collate_config.get('patch_size', 48)

        hr_patches = []
        for img in batch:
            _, H, W = img.shape
            if H < patch or W < patch:
                raise ValueError(f"HR image too small to crop {patch}x{patch}")

            top = random.randint(0, H - patch)
            left = random.randint(0, W - patch)

            crop = img[:, top:top + patch, left:left + patch]
            hr_patches.append(crop)

        hr_tensor = torch.stack(hr_patches, dim=0)

        # optional random scale (train/val only)
        min_s = collate_config.get('min_scale', 1)
        max_s = collate_config.get('max_scale', 4)
        scale = random.randint(min_s, max_s)
        interp_mode = collate_config.get('interp_mode', 'bicubic')
        lr_tensor = F.interpolate(hr_tensor, scale_factor=1/scale, mode=interp_mode, align_corners=False)

        return {
            "hr": hr_tensor,   # HR crop batch
            "lr": lr_tensor,   # LR crop batch
            "scale": scale,
            # "phase": phase
        }

    # ------------------------------
    # 2. test/infer: batch = LR images
    # ------------------------------
    elif phase in ['test', 'infer']:
        lr_list = [item['lr_image'] for item in batch]  # each: C,h,w
        lr_tensor = torch.stack(lr_list, dim=0)

        scale = collate_config.get('scale', 2)

        # 保持和训练/验证接口一致，hr=None
        return {
            "hr": None,
            "lr": lr_tensor,
            "scale": scale,
            # "phase": phase
        }

    else:
        raise ValueError(f"Unknown phase: {phase}")
