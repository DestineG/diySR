import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def _get_data_range(pred, hr):
    """自动计算 data_range = max - min"""
    min_v = min(pred.min(), hr.min())
    max_v = max(pred.max(), hr.max())
    return (max_v - min_v).item()


# =========================================
# PSNR 单张计算
# =========================================
def calc_psnr_single(pred, hr):
    """
    pred, hr: CHW tensor
    返回单张 PSNR
    """
    pred = pred.double().unsqueeze(0)  # CHW -> NCHW
    hr = hr.double().unsqueeze(0)

    data_range = _get_data_range(pred, hr)
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)
    return psnr_metric(pred, hr).item()


# =========================================
# PSNR 列表计算（遍历单张）
# =========================================
def calc_psnr_batch(pred_list, hr_list):
    """
    pred_list / hr_list: list of CHW tensor
    返回 list of PSNR
    """
    return [calc_psnr_single(p, h) for p, h in zip(pred_list, hr_list)]


# =========================================
# SSIM 单张计算
# =========================================
def calc_ssim_single(pred, hr):
    """
    pred, hr: CHW tensor
    返回单张 SSIM
    """
    pred = pred.double().unsqueeze(0)  # CHW -> NCHW
    hr = hr.double().unsqueeze(0)

    data_range = _get_data_range(pred, hr)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)
    return ssim_metric(pred, hr).item()


# =========================================
# SSIM 列表计算（遍历单张）
# =========================================
def calc_ssim_batch(pred_list, hr_list):
    """
    pred_list / hr_list: list of CHW tensor
    返回 list of SSIM
    """
    return [calc_ssim_single(p, h) for p, h in zip(pred_list, hr_list)]


# 单张
pred = torch.rand(3, 256, 256)
hr   = torch.rand(3, 256, 256)

psnr = calc_psnr_single(pred, hr)
ssim = calc_ssim_single(pred, hr)
print(psnr, ssim)

# 多张
pred_list = [torch.rand(3, 256, 256) for _ in range(5)]
hr_list   = [torch.rand(3, 256, 256) for _ in range(5)]

psnrs = calc_psnr_batch(pred_list, hr_list)
ssims = calc_ssim_batch(pred_list, hr_list)
print(psnrs)
print(ssims)
