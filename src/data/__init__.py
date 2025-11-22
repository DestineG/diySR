# /src/data/__init__.py

from functools import partial
from torch.utils.data import DataLoader
from .dataset import CustomDataset
from .collate import get_collate_by_name

dataloader_defaultConfig = {
    'name': 'div2k_hr',
    'dataloader':{
        'num_workers': 4,
        'batch_size': 16,
        'shuffle': True,
    },
    'dataset':{
        'repeat': 1,
        'loader':{
            'loaderFuncName': 'div2k_hr',
            'loaderFuncArgs': {
                'storage_type': 'disk',
                'phase': 'train',
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
        },
        'handler':{
            'handlerFuncName': 'div2k_hr',
            'handlerFuncArgs': {
                'storage_type': 'disk',
                'color_mode': 'RGB',
                'phase': 'train'
            }
        }
    },
    'collate':{
        'collateFuncName': 'div2k_hr',
        'collateFuncArgs': {
            'phase': 'train',
            'patch_size': 128,
            'min_scale': 1,
            'max_scale': 4,
            'interp_mode': 'bicubic'
        }
    }
}
def get_dataloader():
    dataloader_config = dataloader_defaultConfig.get('dataloader', {})
    dataset_config = dataloader_defaultConfig.get('dataset', {})
    dataset = CustomDataset(dataset_config=dataset_config)

    collate_config = dataloader_defaultConfig.get('collate', {})
    collate_func_name = collate_config.get('collateFuncName', 'div2k_hr')
    collate_func_args = collate_config.get('collateFuncArgs', {})
    collate_fn = partial(get_collate_by_name(collate_func_name), collate_config=collate_func_args)

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config.get('batch_size', 16),
        shuffle=dataloader_config.get('shuffle', True),
        num_workers=dataloader_config.get('num_workers', 4),
        collate_fn=collate_fn
    )
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader()
    for batch in dataloader:
        print(batch.keys())
        break