# src/test.py

import yaml
from tqdm import tqdm
import torch

from torchmetrics.functional import peak_signal_noise_ratio as tm_psnr
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

from src.data import get_dataloader
from src.models import get_model_by_name

# -----------------------------
# 1️⃣ 配置读取函数
# -----------------------------
def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    test_data_loader = cfg['test_data_loader']
    model_config = cfg['model']
    return test_data_loader, model_config


# -----------------------------
# 2️⃣ PSNR / SSIM 计算函数
# -----------------------------
def calc_psnr(pred, hr):
    # 转 double 更稳定
    pred = pred.double()
    hr   = hr.double()

    return tm_psnr(pred, hr, data_range=255.0).item()


def calc_ssim(pred, hr):
    pred = pred.double()
    hr   = hr.double()

    return tm_ssim(pred, hr, data_range=255.0).item()


# -----------------------------
# 3️⃣ 测试函数
# -----------------------------
def test(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()

    total_psnr = 0
    total_ssim = 0
    count = 0

    pbar = tqdm(test_loader, desc="Testing", ascii=True)

    for step, batch in enumerate(pbar):
        # batch: lr hr scale
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            # out: decoded
            out = model(batch)   # out = [B,C,H,W] assumed 0~255

        pred = out["decoded"].clamp(0, 255)
        hr = batch["hr"].clamp(0, 255) if batch["hr"] is not None else None

        if hr is not None:
            psnr = calc_psnr(pred, hr)
            ssim = calc_ssim(pred, hr)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            pbar.set_postfix({
                "PSNR": f"{psnr:.2f}",
                "SSIM": f"{ssim:.4f}"
            })

        # 这里可以添加保存结果的代码
        # ...

    # -----------------------------
    # 输出平均指标
    # -----------------------------
    if count > 0:
        print(f"\nAverage PSNR: {total_psnr / count:.2f} dB")
        print(f"Average SSIM: {total_ssim / count:.4f}")
    else:
        print("No HR available → cannot compute PSNR/SSIM.")

    print("Testing completed.")


# -----------------------------
# 4️⃣ 主函数 python -m src.test
# -----------------------------
# Average PSNR: 32.19 dB
# Average SSIM: 0.9147
if __name__ == "__main__":
    yaml_path = './src/configs/test/div2k_hr_baseModel.yaml'
    test_loader_cfg, model_cfg = load_config(yaml_path)

    # dataloader
    test_loader = get_dataloader(test_loader_cfg)

    # model
    model_name = model_cfg.get('modelClsName', 'base_model')
    model_args = model_cfg.get('modelClsArgs', model_cfg)
    model = get_model_by_name(model_name)(model_args)
    model.setup()

    # test
    test(model, test_loader, device='cuda')
