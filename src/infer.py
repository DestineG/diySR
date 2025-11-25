# src/infer.py

import os
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torchvision.utils as vutils

from src.data import get_dataloader
from src.models import get_model_by_name

# -----------------------------
# 1️⃣ 配置读取函数
# -----------------------------
def load_config(yaml_path):
    cfg = OmegaConf.load(yaml_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    infer_data_loader = cfg_dict.get('infer_data_loader')
    model_config = cfg_dict.get('model')
    infer_config = cfg_dict.get('infer_config')

    return infer_data_loader, model_config, infer_config

# -----------------------------
# 3️⃣ 推理函数
# -----------------------------
def infer(model, test_loader, normalize_config=None, device='cuda', output_dir='./figure/output'):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    pbar = tqdm(test_loader, desc="infering", ascii=True)

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model(batch)  # out = [B,C,H,W]

        if normalize_config is not None and normalize_config.get('normalize', False):
            normalize_mean = normalize_config.get('mean', [0.5, 0.5, 0.5])
            normalize_std = normalize_config.get('std', [0.5, 0.5, 0.5])
            mean = torch.tensor(normalize_mean, device=out['decoded'].device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=out['decoded'].device).view(1, -1, 1, 1)
            # 反归一化
            out['decoded'] = (out['decoded'] * std + mean) * 255.0
            if batch["hr"] is not None:
                batch["hr"] = (batch["hr"] * std + mean) * 255.0

        pred = out["decoded"].clamp(0, 255)


        # -----------------------------
        # 保存结果
        # -----------------------------
        for i in range(pred.shape[0]):  # batch_size
            save_path = os.path.join(output_dir, f"pred_{step*pred.shape[0]+i:04d}.png")
            # scale到0~1再保存
            vutils.save_image(pred[i] / 255.0, save_path)

    print(f"Infering completed. Results saved in: {output_dir}")


# -----------------------------
# 4️⃣ 主函数 python -m src.infer
# -----------------------------
if __name__ == "__main__":
    yaml_path = './src/configs/infer/div2k_hr_baseModel.yaml'
    infer_loader_cfg, model_cfg, infer_config = load_config(yaml_path)
    normalize_config = infer_loader_cfg.get('collate').get('collateFuncArgs').get('normalize_config')

    # dataloader
    infer_loader = get_dataloader(infer_loader_cfg)

    # model
    model_name = model_cfg.get('modelClsName')
    model_args = model_cfg.get('modelClsArgs')
    model = get_model_by_name(model_name)(model_args)
    weight_path = infer_config.get('weight_path')
    model.setup(weight_path)
    device = infer_config.get('device')

    # infer
    infer(model, infer_loader, normalize_config=normalize_config, device=device)
