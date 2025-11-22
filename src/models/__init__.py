# src/models/__init__.py

import os
import glob
import importlib
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