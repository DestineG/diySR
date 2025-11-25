# src/train.py

import os
import re
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from src.data import get_dataloader
from src.models import get_model_by_name

# -----------------------------
# 1️⃣ 配置读取函数
# -----------------------------
def load_config(yaml_path):
    cfg = OmegaConf.load(yaml_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    train_data_loader = cfg_dict.get('train_data_loader')
    val_data_loader = cfg_dict.get('val_data_loader')
    model_config = cfg_dict.get('model')
    train_config = cfg_dict.get('train_config')

    return train_data_loader, val_data_loader, model_config, train_config

def get_next_exp_dir(log_root, index_exp=None):
    os.makedirs(log_root, exist_ok=True)

    if index_exp is not None:
        exp_dir = os.path.join(log_root, f"exp_{index_exp}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    exp_dirs = [d for d in os.listdir(log_root) if re.match(r"exp_\d+", d)]
    if not exp_dirs:
        next_id = 1
    else:
        nums = [int(re.findall(r"\d+", d)[0]) for d in exp_dirs]
        next_id = max(nums) + 1

    new_exp_dir = os.path.join(log_root, f"exp_{next_id}")
    os.makedirs(new_exp_dir)
    return new_exp_dir

# -----------------------------
# 2️⃣ 训练函数
# -----------------------------
def train(model, train_loader, val_loader, train_config=None):
    # 从 checkpoint 恢复状态
    resume = train_config.get('resume', False)
    if resume:
        weight_path = train_config.get('weight_path')
    else:
        weight_path = None
    start_epoch, optimizer_state, best_val_loss = model.setup(weight_path)
    device = train_config.get('device')
    model = model.to(device)

    optim_config = train_config['optimizer']
    optim_name = optim_config.get('optimizerClsName')
    optim_args = optim_config.get('optimizerArgs')
    optimizer = getattr(optim, optim_name)(model.parameters(), **optim_args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    scheduler_config = train_config.get('lr_scheduler')
    scheduler_name = scheduler_config.get('schedulerClsName')
    scheduler_args = scheduler_config.get('schedulerArgs')
    scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_args)

    loss_config = train_config.get('loss')
    loss_name = loss_config.get('lossClsName')
    loss_args = loss_config.get('lossArgs')
    criterion = getattr(nn, loss_name)(**loss_args)

    # 自动创建 exp_xx 目录
    exp_root = train_config.get('experiments_root')
    os.makedirs(exp_root, exist_ok=True)
    index_exp = train_config.get('index_exp')
    exp_dir = get_next_exp_dir(exp_root, index_exp)
    print(f"[INFO] Experiment directory: {exp_dir}")
    save_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    num_epochs = train_config.get('max_epochs')
    save_interval = train_config.get('save_interval')
    img_log_step = train_config.get('img_log_step')
    
    global_step = start_epoch * len(train_loader)  # 从 epoch 对应的 step 开始
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ascii=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out['decoded'], batch['hr'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
            
            writer.add_scalar('TrainStepLoss', loss.item(), global_step)
            
            if global_step % img_log_step == 0:
                lr_img = vutils.make_grid(batch['lr'][:4], normalize=True, scale_each=True)
                hr_img = vutils.make_grid(batch['hr'][:4], normalize=True, scale_each=True)
                pred_img = vutils.make_grid(out['decoded'][:4], normalize=True, scale_each=True)
                writer.add_image('Images/LR', lr_img, global_step)
                writer.add_image('Images/Pred', pred_img, global_step)
                writer.add_image('Images/HR', hr_img, global_step)
            
            global_step += 1
        scheduler.step()
        epoch_train_loss = running_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ascii=True):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                out = model(batch)
                val_loss_epoch += criterion(out['decoded'], batch['hr']).item()
        val_loss_epoch /= len(val_loader)
        writer.add_scalars('EpochLoss', {
            'Train': epoch_train_loss,
            'Val': val_loss_epoch
        }, epoch)
        
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            model.save_checkpoint(epoch+1, model.state_dict(), optimizer.state_dict(), val_loss_epoch, save_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

        # 保存最优权重
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            save_path = os.path.join(save_dir, f'best_model.pth')
            model.save_checkpoint(epoch+1, model.state_dict(), optimizer.state_dict(), val_loss_epoch, save_path)
            print(f"Saved best model checkpoint at epoch {epoch+1}, val_loss={val_loss_epoch:.6f}")
    
    writer.close()


# -----------------------------
# 3️⃣ 主函数 python -m src.train
# tensorboard --logdir=./experiments
# -----------------------------
if __name__ == "__main__":
    yaml_path = './src/configs/train/div2k_hr_baseModel.yaml'
    train_loader_cfg, val_loader_cfg, model_cfg, train_config = load_config(yaml_path)

    # 构建 dataloader
    train_loader = get_dataloader(train_loader_cfg)
    val_loader = get_dataloader(val_loader_cfg)

    # 构建模型
    model_name = model_cfg.get('modelClsName')
    model_args = model_cfg.get('modelClsArgs')
    model = get_model_by_name(model_name)(model_args)

    # 启动训练
    train(
        model, 
        train_loader, 
        val_loader, 
        train_config=train_config,
    )
