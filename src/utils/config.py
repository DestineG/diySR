# src/utils/config.py

from omegaconf import DictConfig, OmegaConf


def parse_cfg(cfg: DictConfig):
    # 注册解析器
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    
    # 转成普通 dict，并解析所有引用
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 返回解析后的配置
    return cfg_dict