# /src/data/dataset.py

from torch.utils.data import Dataset

from .loaders import get_loader_by_name
from .handlers import get_handler_by_name


class BaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
        # 数据加载
        loader_config = self.config.get('loader_config')
        loader_func_name = loader_config.get('loaderFuncName')
        loader_func_args = loader_config.get('loaderFuncArgs')
        self.data_loader = get_loader_by_name(loader_func_name)
        self.samples, self.num_samples = self.data_loader(loader_func_args)

        # 数据处理
        handler_config = self.config.get('handler_config')
        handler_func_name = handler_config.get('handlerFuncName')
        self.handler_func_args = handler_config.get('handlerFuncArgs')
        self.data_handler = get_handler_by_name(handler_func_name)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_handler(idx, self.samples, self.handler_func_args)
