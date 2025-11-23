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
        hr_list = []
        lr_list = []
        scale = collate_config.get('scale', 2)
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
