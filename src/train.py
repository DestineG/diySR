# src/train.py

import os
import hydra
from omegaconf import DictConfig

from .utils.config import parse_cfg
from .data import get_datamodule_by_name
from .models import get_model_by_name
from .trainers import get_trainer_by_name


configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
train_configs_dir = os.path.join(configs_dir, 'train')
print(train_configs_dir)
@hydra.main(version_base=None, config_path=train_configs_dir, config_name="config")
def train_app(cfg: DictConfig):
    cfg = parse_cfg(cfg)
    data_config = cfg.get('data')
    model_config = cfg.get('model')
    train_config = cfg.get('train')

    datamodule_cls_name = data_config.get('datamoduleClsName')
    datamodule_cls_args = data_config.get('datamoduleClsArgs')
    DataModuleClass = get_datamodule_by_name(datamodule_cls_name)
    data_module = DataModuleClass(datamodule_cls_args)
    data_module.setup()

    model_cls_name = model_config.get('modelClsName')
    model_cls_args = model_config.get('modelClsArgs')
    ModelClass = get_model_by_name(model_cls_name)
    model = ModelClass(model_cls_args)

    trainer_cls_name = train_config.get('trainerClsName')
    trainer_cls_args = train_config.get('trainerClsArgs')
    TrainerClass = get_trainer_by_name(trainer_cls_name)
    trainer = TrainerClass(model, data_module, trainer_cls_args)
    # trainer.setup()
    # trainer.fit()


# nohup python -m src.demo > a.log &
# tensorboard --logdir=experiments
if __name__ == "__main__":
    train_app()
