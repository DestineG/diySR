# src/models/decoders/__init__.py

import os
import glob
import importlib

registered_decoders = {}
def register_decoder(name):
    def decorator(cls):
        registered_decoders[name] = cls
        return cls
    return decorator

# 根据名称获取已注册的解码器
def get_decoder_by_name(name):
    decoder = registered_decoders.get(name)
    if decoder is None:
        raise ValueError(f"Decoder with name '{name}' is not registered.")
    return decoder

# 自动注册所有解码器
# 自动注册调用放到最后避免循环导入问题
def register_all_decoders():
    current_dir = os.path.dirname(__file__)
    decoder_files = glob.glob(os.path.join(current_dir, "*_decoder.py"))

    for decoder_file in decoder_files:
        module_name = os.path.splitext(os.path.basename(decoder_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_decoders()