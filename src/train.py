# src/train.py

import os
import yaml
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
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    train_data_loader = cfg['train_data_loader']
    val_data_loader = cfg['val_data_loader']
    model_config = cfg['model']
    return train_data_loader, val_data_loader, model_config

# -----------------------------
# 2️⃣ 训练函数
# -----------------------------
def train(model, train_loader, val_loader, device='cuda', num_epochs=100, lr=1e-4, save_dir='./checkpoints', log_dir='./logs', img_log_step=100):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ascii=True)
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out['decoded'], batch['hr'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
            
            # TensorBoard: 每 step 写入训练 loss
            writer.add_scalar('Train/StepLoss', loss.item(), global_step)
            
            # TensorBoard: 每 img_log_step 写一次预测图
            if global_step % img_log_step == 0:
                # 假设 batch['lr'] 是低分图，batch['hr'] 是高分图，out['decoded'] 是预测
                lr_img = vutils.make_grid(batch['lr'][:4], normalize=True, scale_each=True)
                hr_img = vutils.make_grid(batch['hr'][:4], normalize=True, scale_each=True)
                pred_img = vutils.make_grid(out['decoded'][:4], normalize=True, scale_each=True)
                writer.add_image('Input/LR', lr_img, global_step)
                writer.add_image('Output/Pred', pred_img, global_step)
                writer.add_image('Target/HR', hr_img, global_step)
            
            global_step += 1
        
        epoch_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Train/EpochLoss', epoch_train_loss, epoch)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ascii=True):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                out = model(batch)
                val_loss += criterion(out['decoded'], batch['hr']).item()
        val_loss /= len(val_loader)
        writer.add_scalar('Val/EpochLoss', val_loss, epoch)
        print(f"Epoch {epoch+1} validation loss: {val_loss:.6f}")
        
        # 保存最优权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f'best_model.pth')
            model.save_checkpoint(epoch+1, model.state_dict(), optimizer.state_dict(), val_loss, save_path)
            print(f"Saved best model checkpoint at epoch {epoch+1}, val_loss={val_loss:.6f}")
    
    writer.close()

# -----------------------------
# 3️⃣ 主函数
# tensorboard --logdir=./logs
# -----------------------------
if __name__ == "__main__":
    yaml_path = './src/configs/train/div2k_hr_baseModel.yaml'
    train_loader_cfg, val_loader_cfg, model_cfg = load_config(yaml_path)

    # 构建 dataloader
    train_loader = get_dataloader(train_loader_cfg)
    val_loader = get_dataloader(val_loader_cfg)

    # 构建模型
    model_name = model_cfg.get('modelClsName', 'base_model')
    model_args = model_cfg.get('modelClsArgs', model_cfg)
    model = get_model_by_name(model_name)(model_args)
    model.setup()

    # 启动训练
    train(model, train_loader, val_loader, device='cuda', num_epochs=1000, lr=1e-4, save_dir='./checkpoints', log_dir='./logs', img_log_step=200)
