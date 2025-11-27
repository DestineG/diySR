import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="/workspace/projects/dl/diySR/src/configs/train", config_name="config")
def train_app(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)  # 解析插值
    print("Training config:", cfg)

if __name__ == "__main__":
    train_app()
