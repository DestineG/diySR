# /src/data/loaders/__init__.py

import os
import glob
import importlib
import cv2
from tqdm import tqdm

# 初始化 loader 注册器和字典
registered_loaders = {}
def register_loader(name):
    def decorator(func):
        registered_loaders[name] = func
        return func
    return decorator

def get_loader_by_name(name):
    loader = registered_loaders.get(name)
    if loader is None:
        raise ValueError(f"Loader with name '{name}' is not registered.")
    return loader

def _load_image(img_path, color_mode='RGB'):
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
# 自动注册调用放到最后避免循环导入问题
def register_all_loaders():
    current_dir = os.path.dirname(__file__)
    loader_files = glob.glob(os.path.join(current_dir, "*_loader.py"))

    for loader_file in loader_files:
        module_name = os.path.splitext(os.path.basename(loader_file))[0]
        importlib.import_module(f".{module_name}", package=__name__)
register_all_loaders()