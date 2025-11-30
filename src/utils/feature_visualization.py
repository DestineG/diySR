# src/utils/feature_visualization.py

import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def plot_channel_means(fmap: torch.Tensor, global_step: int, writer: SummaryWriter, tag="ChannelMeans"):
    """
    fmap: C x H x W
    """
    # 计算每个通道的空间平均值
    channel_means = fmap.detach().mean(dim=(1,2)).cpu().numpy()

    # 绘图
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(channel_means, marker='.')
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean Activation")
    ax.grid(True)

    # 写入 TensorBoard
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)
