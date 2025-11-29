# /src/data/loaders/div2k_hr_loader.py

from . import register_loader, _load_images_to_memory
from ..splitters import get_splitter_by_name


@register_loader('div2k_hr')
def div2k_hr_loader(config=None):
    # 数据拆分
    split_config = config.get('split_config')
    split_func_name = split_config.get('splitFuncName')
    split_func_args = split_config.get('splitFuncArgs')
    split_constructor = get_splitter_by_name(split_func_name)
    if split_constructor is None:
        raise RuntimeError(f"Split function '{split_func_name}' not found.")
    split_info = split_constructor(split_func_args)

    # 数据加载
    mode = config.get('mode')
    load_to = config.get('load_to')
    repeat = config.get('repeat')

    if load_to == 'disk':
        paths = split_info.get(f'{mode}_paths')
        num_samples = split_info.get(f'num_{mode}')

        return paths, num_samples * repeat

    elif load_to == 'memory':
        paths = split_info.get(f'{mode}_paths')
        num_samples = split_info.get(f'num_{mode}')
        load_to_memory_config = config.get('load_to_memory_config')

        images = _load_images_to_memory(
            paths,
            max_memory_usage=load_to_memory_config.get('max_memory_usage'),
            color_mode=load_to_memory_config.get('color_mode', 'RGB'),
            verbose=load_to_memory_config.get('verbose')
        )

        return images, num_samples * repeat
    else:
        raise ValueError(f"Unknown load_to type: {load_to}")
