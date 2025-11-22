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
    """
    根据名称返回已注册的解码器类
    """
    decoder = registered_decoders.get(name)
    if decoder is None:
        raise ValueError(f"Decoder with name '{name}' is not registered.")
    return decoder

# 自动注册所有解码器
def register_all_decoders():
    """
    自动扫描当前目录下所有 *_decoder.py 文件并 import
    触发 @register_decoder 装饰器自动注册
    """

    current_dir = os.path.dirname(__file__)
    decoder_files = glob.glob(os.path.join(current_dir, "*_decoder.py"))

    for decoder_file in decoder_files:
        module_name = os.path.splitext(os.path.basename(decoder_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_decoders()