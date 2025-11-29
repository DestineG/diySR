# src/utils/config.py

from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any


def parse_cfg(cfg: DictConfig) -> Dict[str, Any]:
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    return OmegaConf.to_container(cfg, resolve=True)


def parse_cfg_from_yaml(yaml_path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(yaml_path)
    return parse_cfg(cfg)


def save_cfg_to_yaml(cfg: Dict[str, Any], yaml_path: str) -> None:
    cfg_omega = OmegaConf.create(cfg)
    OmegaConf.save(cfg_omega, yaml_path)
