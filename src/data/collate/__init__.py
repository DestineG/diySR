# /src/data/collate/__init__.py

import os
import glob
import importlib

registered_collates = {}
def register_collate(name):
    def decorator(func):
        registered_collates[name] = func
        return func
    return decorator

# 获取已注册的collate
def get_collate_by_name(name):
    """
    根据名称返回已注册的collate
    """
    collate = registered_collates.get(name)
    if collate is None:
        raise ValueError(f"Collate with name '{name}' is not registered.")
    return collate

# 自动注册所有 collate
def register_all_collates():
    """
    自动扫描当前目录下所有 *_collate.py 文件并 import
    触发 @register_collate 装饰器自动注册
    """
    current_dir = os.path.dirname(__file__)
    collate_files = glob.glob(os.path.join(current_dir, "*_collate.py"))

    for collate_file in collate_files:
        module_name = os.path.splitext(os.path.basename(collate_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_collates()