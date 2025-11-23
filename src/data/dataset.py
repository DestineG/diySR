# /src/data/dataset.py

import torch
from torch.utils.data import Dataset

from .loaders import get_loader_by_name
from .handlers import get_handler_by_name

dataset_defaultConfig = {
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
            'phase': 'train',
            'repeat': True
        }
    }
}
class CustomDataset(Dataset):
    def __init__(self, dataset_config=dataset_defaultConfig):
        self.dataset_config = dataset_config
        # 数据加载
        self.loader = self.dataset_config.get('loader', {})
        loader_name = self.loader.get('loaderFuncName', 'div2k_hr')
        loader_config = self.loader.get('loaderFuncArgs', {})
        self.data_loader = get_loader_by_name(loader_name)
        self.samples, self.num_samples = self.data_loader(loader_config)

        # 数据处理
        self.handler = self.dataset_config.get('handler', {})
        handler_name = self.handler.get('handlerFuncName', 'div2k_hr')
        self.handler_config = self.handler.get('handlerFuncArgs', {})
        self.data_handler = get_handler_by_name(handler_name)

    def __len__(self):
        return len(self.samples)*self.dataset_config.get('repeat', 1)

    def __getitem__(self, idx):
        return self.data_handler(idx, self.samples, self.handler_config)


import unittest

# 假设所有相关模块已经正确导入，并且 get_loader_by_name 和 get_handler_by_name 已经实现

class TestCustomDataset(unittest.TestCase):
    
    def setUp(self):
        """
        在每个测试前运行，初始化自定义数据集。
        """
        # 使用默认配置初始化 CustomDataset
        self.dataset = CustomDataset(dataset_config=dataset_defaultConfig)

    def test_dataset_length(self):
        """
        测试数据集的长度，确保 __len__ 方法返回正确的样本数。
        """
        length = len(self.dataset)
        self.assertGreater(length, 0, "Dataset length should be greater than 0")

    def test_get_item(self):
        """
        测试 __getitem__ 方法，确保它返回正确的数据。
        """
        idx = 0  # 假设至少有一个样本
        sample = self.dataset[idx]
        self.assertIsNotNone(sample, f"Sample at index {idx} should not be None")
        self.assertIsInstance(sample, torch.Tensor, "Sample should be a torch Tensor")

    def test_loader_function(self):
        """
        测试加载器函数是否正确执行，确保 get_loader_by_name 能返回有效的加载器。
        """
        loader = self.dataset.data_loader
        self.assertTrue(callable(loader), "Loader function should be callable")

    def test_handler_function(self):
        """
        测试数据处理器是否正确执行，确保 get_handler_by_name 返回有效的处理器。
        """
        handler = self.dataset.data_handler
        self.assertTrue(callable(handler), "Handler function should be callable")


# python -m unittest src.data
if __name__ == "__main__":
    unittest.main()
