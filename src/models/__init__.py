# src/models/__init__.py

import os
import glob
import importlib
import torch
import torch.nn as nn

registered_models = {}
def register_model(name):
    def decorator(cls):
        registered_models[name] = cls
        return cls
    return decorator

# 根据名称获取已注册的model
def get_model_by_name(name):
    """
    根据名称返回已注册的model类
    """
    model = registered_models.get(name)
    if model is None:
        raise ValueError(f"Model with name '{name}' is not registered.")
    return model

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")
    
    def save_checkpoint(self, epoch, model_state_dict, optimizer_state_dict, val_loss, path):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'val_loss': val_loss
            }, path)

    def setup(self):
        weight_path = self.config.get('weight_path', None)
        if weight_path is not None and os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            optim_state = checkpoint.get('optimizer_state_dict', None)
            val_loss = checkpoint.get('val_loss', float('inf'))

            return epoch, optim_state, val_loss
        else:
            return 0, None, float('inf')

# 自动注册所有model
def register_all_models():
    """
    自动扫描当前目录下所有 *_model.py 文件并 import
    触发 @register_model 装饰器自动注册
    """

    current_dir = os.path.dirname(__file__)
    model_files = glob.glob(os.path.join(current_dir, "*_model.py"))

    for model_file in model_files:
        module_name = os.path.splitext(os.path.basename(model_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_models()