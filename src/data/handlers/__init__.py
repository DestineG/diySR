# /src/data/handlers/__init__.py

import os
import glob
import importlib

# 初始化 handler 注册器
registered_handlers = {}

def register_handler(name):
    """
    装饰器函数，用于注册数据处理器
    """
    def decorator(func):
        registered_handlers[name] = func
        return func
    return decorator

# 自动注册所有 handler
def register_all_handlers():
    """
    自动扫描当前目录下所有 *_handler.py 文件并 import
    触发 @register_handler 装饰器自动注册
    """
    current_dir = os.path.dirname(__file__)
    handler_files = glob.glob(os.path.join(current_dir, "*_handler.py"))

    for handler_file in handler_files:
        module_name = os.path.splitext(os.path.basename(handler_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_handlers()

def get_handler_by_name(name):
    """
    根据名称返回已注册的处理器
    """
    handler = registered_handlers.get(name)
    if handler is None:
        raise ValueError(f"Handler with name '{name}' is not registered.")
    return handler
