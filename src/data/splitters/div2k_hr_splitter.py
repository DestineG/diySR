# /src/data/splitters/div2k_hr_splitter.py

import os
import random
import yaml

from . import register_splitter, split_results_dir


@register_splitter("div2k_hr")
def split_data_div2k_hr(config):

    data_path = config.get("data_path")
    train_ratio = config.get("train_ratio", 0.9)
    seed = config.get("seed", 42)

    train_path = os.path.join(data_path, "DF2K_train_HR")
    test_path = os.path.join(data_path, "DIV2K_valid_HR")

    train_data = sorted(os.listdir(train_path))
    test_data = sorted(os.listdir(test_path))

    # 自动生成 YAML 保存目录
    yaml_dir = split_results_dir
    os.makedirs(yaml_dir, exist_ok=True)

    # YAML 文件名固定，只使用随机种子
    yaml_filename = f"{split_data_div2k_hr.__name__}_seed{seed}.yaml"
    yaml_path = os.path.join(yaml_dir, yaml_filename)

    # 如果 YAML 已存在，直接读取
    if os.path.exists(yaml_path):
        print(f"YAML file already exists, loading: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            split_info = yaml.safe_load(f)
        return split_info

    # 设置随机种子并划分训练/验证集
    rnd = random.Random(seed)
    rnd.shuffle(train_data)
    split_index = int(len(train_data) * train_ratio)
    train_data_split, val_data_split = train_data[:split_index], train_data[split_index:]

    split_train_path = [os.path.join(train_path, x) for x in train_data_split]
    split_val_path = [os.path.join(train_path, x) for x in val_data_split]
    test_data_path = [os.path.join(test_path, x) for x in test_data]

    # 用字典标记哪些集合存在
    split_type = {
        "train": bool(split_train_path),
        "val": bool(split_val_path),
        "test": bool(test_data_path)
    }

    split_info = {
        "split_type": split_type,
        "random_seed": seed,
        "train_ratio": train_ratio,
        "num_train": len(split_train_path),
        "num_val": len(split_val_path),
        "num_test": len(test_data_path),
        "train_paths": split_train_path,
        "val_paths": split_val_path,
        "test_paths": test_data_path
    }

    # 保存 YAML
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(split_info, f, sort_keys=False, allow_unicode=True)
    print(f"Split information has been saved to {yaml_path}")

    return split_info


if __name__ == "__main__":
    data_path = "/dataroot/liujiang/data/datasets/DF2K"
    split_info = split_data_div2k_hr(data_path)
    print(split_info)
