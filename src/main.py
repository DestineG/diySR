# src/main.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
def train(model, train_loader, val_loader, device='cuda', num_epochs=100, lr=1e-4, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ascii=True)
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out['decoded'], batch['hr'])
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ascii=True):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                out = model(batch)
                val_loss += criterion(out['decoded'], batch['hr']).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} validation loss: {val_loss:.6f}")
        
        # 保存最优权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print(f"Saved best model checkpoint at epoch {epoch+1}, val_loss={val_loss:.6f}")

# -----------------------------
# 3️⃣ 主函数
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

    # 启动训练
    train(model, train_loader, val_loader, device='cuda', num_epochs=1000, lr=1e-4, save_dir='./checkpoints')
