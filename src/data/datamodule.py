# src/data/datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import partial
from .dataset import BaseDataset
from .collate import get_collate_by_name


registered_datamodules = {}
def register_datamodule(name):
    def decorator(cls):
        registered_datamodules[name] = cls
        return cls
    return decorator
def get_datamodule_by_name(name):
    return registered_datamodules.get(name)

@register_datamodule('base_datamodule')
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # 配置加载
        self.train_config = config.get('train_config')
        self.val_config = config.get('val_config')
        self.test_config = config.get('test_config')

        # 组装 collate 函数
        self.train_collate_config = self.train_config.get('collate_config')
        train_collate_func_name = self.train_collate_config.get('collateFuncName')
        train_collate_func_args = self.train_collate_config.get('collateFuncArgs')
        self.train_collate = partial(
            get_collate_by_name(train_collate_func_name),
            config=train_collate_func_args
        )

        self.val_collate_config = self.val_config.get('collate_config')
        val_collate_func_name = self.val_collate_config.get('collateFuncName')
        val_collate_func_args = self.val_collate_config.get('collateFuncArgs')
        self.val_collate = partial(
            get_collate_by_name(val_collate_func_name),
            config=val_collate_func_args
        )

        self.test_collate_config = self.test_config.get('collate_config')
        test_collate_func_name = self.test_collate_config.get('collateFuncName')
        test_collate_func_args = self.test_collate_config.get('collateFuncArgs')
        self.test_collate = partial(
            get_collate_by_name(test_collate_func_name),
            config=test_collate_func_args
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset_config = self.train_config.get('dataset_config')
            val_dataset_config = self.val_config.get('dataset_config')
            test_dataset_config = self.test_config.get('dataset_config')
            self.train_dataset = BaseDataset(train_dataset_config)
            self.val_dataset = BaseDataset(val_dataset_config)
            self.test_dataset = BaseDataset(test_dataset_config)

        elif stage == 'test':
            test_dataset_config = self.test_config.get('dataset_config')
            self.test_dataset = BaseDataset(test_dataset_config)
        
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        batch_size = self.train_config.get('batch_size')
        num_workers = self.train_config.get('num_workers')
        shuffle = self.train_config.get('shuffle')
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,    
            shuffle=shuffle,
            collate_fn=self.train_collate
        )
    
    def val_dataloader(self):
        batch_size = self.val_config.get('batch_size')
        num_workers = self.val_config.get('num_workers')
        shuffle = self.val_config.get('shuffle')
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,    
            shuffle=shuffle,
            collate_fn=self.val_collate
        )
    
    def test_dataloader(self):
        batch_size = self.test_config.get('batch_size')
        num_workers = self.test_config.get('num_workers')
        shuffle = self.test_config.get('shuffle')
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,    
            shuffle=shuffle,
            collate_fn=self.test_collate
        )
    
    def get_dataloaders(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
