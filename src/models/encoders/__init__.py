# src/models/encoders/__init__.py

import os
import glob
import importlib

registered_encoders = {}
def register_encoder(name):
    def decorator(cls):
        registered_encoders[name] = cls
        return cls
    return decorator

# 根据名称获取已注册的编码器
def get_encoder_by_name(name):
    encoder = registered_encoders.get(name)
    if encoder is None:
        raise ValueError(f"Encoder with name '{name}' is not registered.")
    return encoder

# 自动注册所有编码器
# 自动注册调用放到最后避免循环导入问题
def register_all_encoders():
    current_dir = os.path.dirname(__file__)
    encoder_files = glob.glob(os.path.join(current_dir, "*_encoder.py"))

    for encoder_file in encoder_files:
        module_name = os.path.splitext(os.path.basename(encoder_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_encoders()