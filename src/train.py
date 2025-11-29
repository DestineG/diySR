# src/train.py

import os
import hydra
from omegaconf import DictConfig

from .utils.config import parse_cfg, save_cfg_to_yaml, parse_cfg_from_yaml
from .data import get_datamodule_by_name
from .models import get_model_by_name
from .trainers import get_trainer_by_name


def train(cfg: dict):
    data_config = cfg.get('data')
    model_config = cfg.get('model')
    train_config = cfg.get('train')

    # 加载 DataModule
    datamodule_cls_name = data_config.get('datamoduleClsName')
    datamodule_cls_args = data_config.get('datamoduleClsArgs')
    DataModuleClass = get_datamodule_by_name(datamodule_cls_name)
    data_module = DataModuleClass(datamodule_cls_args)
    data_module.setup(stage='fit')

    # 加载 Model
    model_cls_name = model_config.get('modelClsName')
    model_cls_args = model_config.get('modelClsArgs')
    ModelClass = get_model_by_name(model_cls_name)
    model = ModelClass(model_cls_args)

    # 加载 Trainer
    trainer_cls_name = train_config.get('trainerClsName')
    trainer_cls_args = train_config.get('trainerClsArgs')
    TrainerClass = get_trainer_by_name(trainer_cls_name)
    trainer = TrainerClass(model, data_module, trainer_cls_args)
    trainer.setup(stage='fit')

    # 保存配置文件到Hydra指定目录
    hydra_save_dir = trainer_cls_args.get("experiment_config").get("hydra_config").get("hydra_save_dir")
    hydra_save_path = os.path.join(hydra_save_dir, "config.yaml")
    save_cfg_to_yaml(cfg, hydra_save_path)
    trainer.fit()

configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
train_configs_dir = os.path.join(configs_dir, 'train')
print(train_configs_dir)
@hydra.main(version_base=None, config_path=train_configs_dir, config_name="config")
def train_from_hydraConf(cfg: DictConfig):
    cfg = parse_cfg(cfg)
    train(cfg)

def train_from_yaml(cfg_path: str):
    cfg = parse_cfg_from_yaml(cfg_path)
    train(cfg)

# nohup python -m src.train > train.log &
# tensorboard --logdir=experiments
if __name__ == "__main__":
    train_from_hydraConf()
