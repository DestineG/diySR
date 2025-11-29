# src/test.py

import os

from .utils.config import parse_cfg_from_yaml
from .data import get_datamodule_by_name
from .models import get_model_by_name
from .trainers import get_trainer_by_name


def test(cfg_path: str):
    # 解析配置
    cfg = parse_cfg_from_yaml(cfg_path)
    data_config = cfg.get('data')
    model_config = cfg.get('model')
    train_config = cfg.get('train')
    print(data_config)
    print(model_config)
    print(train_config)

    # 加载 DataModule
    datamodule_cls_name = data_config.get('datamoduleClsName')
    datamodule_cls_args = data_config.get('datamoduleClsArgs')
    DataModuleClass = get_datamodule_by_name(datamodule_cls_name)
    data_module = DataModuleClass(datamodule_cls_args)
    data_module.setup(stage='test')

    # 加载 Model
    model_cls_name = model_config.get('modelClsName')
    model_cls_args = model_config.get('modelClsArgs')
    ModelClass = get_model_by_name(model_cls_name)
    model = ModelClass(model_cls_args)
    experiment_config = train_config.get('trainerClsArgs').get("experiment_config")
    experiments_dir = experiment_config.get("experiment_dir")
    experiment_name = experiment_config.get("experiment_name")
    checkpoint_dir = experiment_config.get("checkpoint_config").get("checkpoint_dir")
    weight_path = os.path.join(
        experiments_dir,
        experiment_name,
        checkpoint_dir,
        "model_epoch_best.pth"
    )
    model.load_weights(weight_path)

    # 加载 Trainer
    trainer_cls_name = train_config.get('trainerClsName')
    trainer_cls_args = train_config.get('trainerClsArgs')
    TrainerClass = get_trainer_by_name(trainer_cls_name)
    trainer = TrainerClass(model, data_module, trainer_cls_args)
    trainer.setup(stage='test')


# nohup python -m src.test > test.log
if __name__ == "__main__":
    test("/workspace/projects/dl/diySR/experiments/exp_01/hydra/config.yaml")
