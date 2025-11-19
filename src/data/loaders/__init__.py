# /src/data/loaders/__init__.py

import os
import glob
import importlib
import cv2
from tqdm import tqdm

# 初始化 loader 注册器和字典
registered_loaders = {}
def register_loader(name):
    """
    装饰器函数，用于注册数据加载器
    """
    def decorator(func):
        registered_loaders[name] = func
        return func
    return decorator

def _load_image(img_path, color_mode='RGB'):
    """
    加载单张图像，支持返回 RGB 或 灰度图像
    
    Args:
        img_path (str): 图像路径
        color_mode (str): 'RGB' 或 'GRAY'，指定返回图像的颜色模式
    
    Returns:
        numpy.ndarray: 加载后的图像（RGB 或 灰度）
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[WARN] Failed to load: {img_path}")
    
    if color_mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid color_mode. Choose 'RGB' or 'GRAY'.")
    
    return img

def _load_images_to_memory(image_paths, max_memory_usage=None, color_mode='RGB', verbose=True):
    """
    将图像加载到内存中
    
    Args:
        image_paths (list): 图像路径列表
        max_memory_usage (float): 最大内存使用量（MB）
        color_mode (str): 'RGB' 或 'GRAY'
        verbose (bool): 是否显示进度条和日志
    
    Returns:
        list: 加载的图像列表
    """
    images = []
    total_memory = 0.0
    iter_paths = tqdm(image_paths, desc="Loading images", ascii=True) if verbose else image_paths

    for img_path in iter_paths:
        img = _load_image(img_path, color_mode)
        if img is None:
            continue
        
        img_memory = img.nbytes / (1024 * 1024)

        if max_memory_usage and (total_memory + img_memory) > max_memory_usage:
            if verbose:
                print(f"[INFO] Memory limit reached at {total_memory:.2f} MB")
            break

        images.append(img)
        total_memory += img_memory

    if verbose:
        print(f"[INFO] Loaded {len(images)} images, memory={total_memory:.2f} MB")

    return images

# 自动注册所有 loader
def register_all_loaders():
    """
    自动扫描当前目录下所有 *_loader.py 文件并 import
    触发 @register_loader 装饰器自动注册
    """
    # 当前目录
    current_dir = os.path.dirname(__file__)
    # 匹配 *_loader.py 文件
    loader_files = glob.glob(os.path.join(current_dir, "*_loader.py"))

    for loader_file in loader_files:
        # 获取模块名
        module_name = os.path.splitext(os.path.basename(loader_file))[0]
        # import 模块（相对于 loaders 包）
        importlib.import_module(f".{module_name}", package=__name__)
register_all_loaders()

def get_loader_by_name(name):
    loader = registered_loaders.get(name)
    if loader is None:
        raise ValueError(f"Loader with name '{name}' is not registered.")
    return loader
