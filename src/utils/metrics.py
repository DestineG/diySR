# src/utils/metrics.py

from typing import List, Dict, Union, Tuple
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def compute_data_range_minmax(normalize_config: Dict) -> Tuple[float, float]:
    """
    根据 normalize_config 计算归一化后像素范围 (min, max)
    normalize_config: dict 包含
        - normalize: bool
        - mean: list of floats
        - std: list of floats
    返回 (min_val, max_val)
    """
    normalize = normalize_config.get("normalize", False)
    mean: List[float] = normalize_config.get("mean", [0.0, 0.0, 0.0])
    std: List[float] = normalize_config.get("std", [1.0, 1.0, 1.0])

    if not normalize:
        # 未归一化，假设像素 0-255
        return 0.0, 255.0

    # 对每个通道计算归一化后最小值和最大值
    min_vals = [(0.0 - m) / s for m, s in zip(mean, std)]
    max_vals = [(1.0 - m) / s for m, s in zip(mean, std)]

    min_val = min(min_vals)
    max_val = max(max_vals)
    return min_val, max_val

def psnr_metric_from_list(
    preds: List[Tensor], 
    targets: List[Tensor], 
    data_range: Union[float, Tuple[float, float]]=None
):
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(preds[0].device)
    for pred, target in zip(preds, targets):
        psnr.update(pred, target)
    return psnr.compute()

def ssim_metric_from_list(
    preds: List[Tensor], 
    targets: List[Tensor], 
    data_range: Union[float, Tuple[float, float]]=None
):
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(preds[0].device)
    for pred, target in zip(preds, targets):
        ssim.update(pred, target)
    return ssim.compute()
