# /src/data/splitters/__init__.py

import os
import glob
import importlib


# 分割结果保存目录
split_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits")

# 初始化 loader 注册器和字典
registered_splitters = {}
def register_splitter(name):
    def decorator(func):
        registered_splitters[name] = func
        return func
    return decorator

def get_splitter_by_name(name):
    splitter = registered_splitters.get(name)
    if splitter is None:
        raise ValueError(f"Splitter with name '{name}' is not registered.")
    return splitter

# 自动注册所有 splitter
# 自动注册调用放到最后避免循环导入问题
def register_all_splitters():
    current_dir = os.path.dirname(__file__)
    splitter_files = glob.glob(os.path.join(current_dir, "*_splitter.py"))

    for splitter_file in splitter_files:
        module_name = os.path.splitext(os.path.basename(splitter_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_splitters()
