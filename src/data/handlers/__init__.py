# /src/data/handlers/__init__.py

import os
import glob
import importlib

# 初始化 handler 注册器
registered_handlers = {}

def register_handler(name):
    def decorator(func):
        registered_handlers[name] = func
        return func
    return decorator

def get_handler_by_name(name):
    handler = registered_handlers.get(name)
    if handler is None:
        raise ValueError(f"Handler with name '{name}' is not registered.")
    return handler

# 自动注册所有 handler
# 自动注册调用放到最后避免循环导入问题
def register_all_handlers():
    current_dir = os.path.dirname(__file__)
    handler_files = glob.glob(os.path.join(current_dir, "*_handler.py"))

    for handler_file in handler_files:
        module_name = os.path.splitext(os.path.basename(handler_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_handlers()