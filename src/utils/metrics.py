# src/utils/metrics.py

from typing import List, Union, Tuple
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


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
