# /src/data/collate/div2k_hr_collate.py

import torch
import torch.nn.functional as F
import random
from . import register_collate

div2k_hr_defaultConfig = {
    'phase': 'train',       # train / val / test_hr / test_lrhr / infer
    'patch_size': 128,
    'min_scale': 1,
    'max_scale': 4,
    'scale': 2,
    'interp_mode': 'bicubic'
}

@register_collate('div2k_hr')
def div2k_hr_collate(batch, collate_config=div2k_hr_defaultConfig):
    phase = collate_config.get('phase', 'train')
    interp_mode = collate_config.get('interp_mode', 'bicubic')

    # ======================================================
    # 1 ⬇️ Training / Validation（输入是 HR，做随机 crop）
    # ======================================================
    if phase in ['train', 'val']:
        patch = collate_config.get('patch_size', 48)

        hr_patches = []
        for img in batch:
            _, H, W = img.shape
            top = random.randint(0, H - patch)
            left = random.randint(0, W - patch)
            crop = img[:, top:top + patch, left:left + patch]
            hr_patches.append(crop)

        hr_tensor = torch.stack(hr_patches, dim=0)

        # 随机 scale
        min_s = collate_config.get('min_scale', 1)
        max_s = collate_config.get('max_scale', 4)
        scale = random.randint(min_s, max_s)

        lr_tensor = F.interpolate(
            hr_tensor, scale_factor=1/scale, mode=interp_mode, align_corners=False
        )

        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "scale": scale,
        }

    # ======================================================
    # 2 ⬇️ Test: HR-only（你给 batch = HR 堆叠 → 需要下采样成 LR）
    # ======================================================
    elif phase == 'test':  
        hr_tensor = torch.stack(batch, dim=0)
        scale = collate_config.get('scale', 2)

        lr_tensor = F.interpolate(
            hr_tensor, scale_factor=1/scale, mode=interp_mode, align_corners=False
        )

        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "scale": scale,
        }

    # ======================================================
    # 3 ⬇️ Inference: 只推理 LR，batch = LR 堆叠
    # ======================================================
    elif phase == 'infer':
        lr_tensor = torch.stack(batch, dim=0)
        scale = collate_config.get('scale', 2)

        return {
            "hr": None,
            "lr": lr_tensor,
            "scale": scale,
        }

    else:
        raise ValueError(f"Unknown phase: {phase}")
