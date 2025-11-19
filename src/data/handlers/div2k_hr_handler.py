# /src/data/handlers.py/div2k_hr_handler.py

import torch

from . import register_handler
from src.data.loaders import _load_image

# 默认配置
div2k_hr_defaultConfig = {
    'storage_type': 'disk',
    'color_mode': 'RGB',
    'phase': 'train'
}
@register_handler('div2k_hr')
def div2k_hr_handler(idx, samples, handler_config=div2k_hr_defaultConfig):
    """
    div2k_hr 数据集处理器
    """
    data = samples[idx]

    storage_type = handler_config.get('storage_type', 'disk')
    if storage_type == 'disk':
        data = _load_image(data, color_mode=handler_config.get('color_mode', 'RGB'))
    data = torch.from_numpy(data).float().permute(2, 0, 1)

    phase = handler_config.get('phase', 'train')
    if phase == 'train':
        return data
    
    elif phase in ['val', 'test']:
        return data

    elif phase == 'infer':
        return data

    else:
        raise ValueError(f"Unknown phase: {phase}")
