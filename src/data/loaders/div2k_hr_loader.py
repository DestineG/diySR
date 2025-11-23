# /src/data/loaders/div2k_hr_loader.py

import os
import gc
import copy
from . import register_loader, _load_images_to_memory
from src.utils.data_split import registered_splits

# 默认配置
div2k_hr_defaultConfig = {
    'storage_type': 'disk',
    'phase': 'train',
    'inference_dir': None,
    'load_to_memory_config': {
        'max_memory_usage': None,  # MB
        'color_mode': 'RGB',
        'verbose': True
    },
    'split': {
        'splitFuncName': 'div2k_hr',
        'splitFuncArgs': {
            'data_path': '/dataroot/liujiang/data/datasets/DF2K',
            'train_ratio': 0.9,
            'seed': 42
        }
    }
}
# 注册 div2k_hr 数据加载器
@register_loader('div2k_hr')
def div2k_hr_loader(loader_config=None):
    """
    div2k_hr 数据集加载器
    """
    loader_config = copy.deepcopy(loader_config or div2k_hr_defaultConfig)

    phase = loader_config.get('phase', 'train')
    storage_type = loader_config.get('storage_type', 'disk')
    load_to_memory_config = loader_config.get('load_to_memory_config', {})

    if phase in ['train', 'val', 'test']:
        split_func_name = loader_config['split']['splitFuncName']
        split_func_args = loader_config['split']['splitFuncArgs']
        split_constructor = registered_splits.get(split_func_name)
        if split_constructor is None:
            raise RuntimeError(f"Split function '{split_func_name}' not found.")
        split_data = split_constructor(**split_func_args)

        split_type = split_data.get('split_type', {})
        if split_type == {} and not split_type.get(phase, False):
            raise ValueError(f"Split type for phase '{phase}' is not defined or empty.")

        if phase+'_paths' not in split_data:
            raise ValueError(f"{phase} split not found in split_data.")

        if storage_type == 'disk':
            return split_data[f'{phase}_paths'], split_data[f'num_{phase}']
        elif storage_type == 'memory':
            valid_paths = [p for p in split_data[f'{phase}_paths'] if os.path.exists(p)]
            if not valid_paths:
                raise ValueError(f"No valid image paths found for {phase}.")

            return _load_images_to_memory(
                valid_paths,
                max_memory_usage=load_to_memory_config.get('max_memory_usage'),
                color_mode=load_to_memory_config.get('color_mode', 'RGB'),
                verbose=load_to_memory_config.get('verbose', True)
            ), split_data[f'num_{phase}']
        else:
            raise ValueError(f"Unknown storage_type: {storage_type}")

    elif phase == 'infer':
        inference_dir = loader_config.get('inference_dir', None)
        if inference_dir is None:
            raise ValueError("inference_dir must be specified for inference phase.")
        infer_paths = sorted([
            os.path.join(inference_dir, p)
            for p in os.listdir(inference_dir)
            if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        if storage_type == 'disk':
            return infer_paths
        elif storage_type == 'memory':
            return _load_images_to_memory(
                infer_paths,
                max_memory_usage=load_to_memory_config.get('max_memory_usage'),
                color_mode=load_to_memory_config.get('color_mode', 'RGB'),
                verbose=load_to_memory_config.get('verbose', True)
            ), len(infer_paths)
        else:
            raise ValueError(f"Unknown storage_type: {storage_type}")
    else:
        raise ValueError(f"Unknown phase: {phase}")

def test_div2k_hr_loader(storage_type='disk', max_memory_usage=20000):
    if storage_type == 'disk':
        print("Testing div2k_hr_loader with disk storage...")
        config = copy.deepcopy(div2k_hr_defaultConfig)
        data = div2k_hr_loader(config)
        print(f"Disk data count: {len(data)}")
    
    elif storage_type == 'memory':
        print("\nTesting div2k_hr_loader with memory storage...")
        config = copy.deepcopy(div2k_hr_defaultConfig)
        config['storage_type'] = 'memory'
        config['load_to_memory_config']['max_memory_usage'] = max_memory_usage  # MB

        try:
            data = div2k_hr_loader(config)
            print(f"Memory data count: {len(data)}")

            # 清理内存
            del data
            gc.collect()
            print("Memory cleared.")

        except Exception as e:
            print(f"Memory loading error: {e}")
    else:
        print("Invalid storage type specified. Choose either 'disk' or 'memory'.")

def main():
    test_div2k_hr_loader(storage_type='disk')
    test_div2k_hr_loader(storage_type='memory', max_memory_usage=20000)

# 调用示例
if __name__ == '__main__':
    main()
