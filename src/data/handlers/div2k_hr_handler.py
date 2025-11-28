# /src/data/handlers.py/div2k_hr_handler.py

import torch

from . import register_handler
from src.data.loaders import _load_image


@register_handler('div2k_hr')
def div2k_hr_handler(idx, samples, config=None):
    load_from = config.get('load_from')
    color_mode = config.get('color_mode')

    # augment_config = config.get('augment_config')
    # augment = augment_config.get('augment')
    # augment_func_name = augment_config.get('augmentFuncName')
    # augment_func_args = augment_config.get('augmentFuncArgs')

    if load_from == 'disk':
        data = _load_image(samples[idx % len(samples)], color_mode=color_mode)
    elif load_from == 'memory':
        data = samples[idx % len(samples)]
    else:
        raise ValueError(f"Unknown load_from: {load_from}")
    
    data = torch.from_numpy(data).float().permute(2, 0, 1)
    return data
