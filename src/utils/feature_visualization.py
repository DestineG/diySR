# src/utils/feature_visualization.py

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def plot_channel_stats(fmap: torch.Tensor, global_step: int, writer: SummaryWriter, tag="ChannelStats"):
    """
    fmap: C x H x W
    """
    # 计算每个通道的空间平均值和标准差
    fmap_detached = fmap.detach()
    channel_means = fmap_detached.mean(dim=(1, 2)).cpu().numpy()
    channel_stds = fmap_detached.std(dim=(1, 2)).cpu().numpy()

    # 绘图：两个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：均值
    axes[0].plot(channel_means, marker='.')
    axes[0].set_title("Channel Means")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Mean")
    axes[0].grid(True)

    # 右图：标准差
    axes[1].plot(channel_stds, marker='.', color='orange')
    axes[1].set_title("Channel Stds")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Std")
    axes[1].grid(True)

    # 写入 TensorBoard
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)