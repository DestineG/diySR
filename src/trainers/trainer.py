# src/trainers/trainer.py

import os
import time
import shutil
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

from .components import get_train_components
from ..utils.metrics import calc_psnr_batch, calc_ssim_batch


registered_trainers = {}
def register_trainer(name):
    def decorator(cls):
        registered_trainers[name] = cls
        return cls
    return decorator

def get_trainer_by_name(name):
    return registered_trainers.get(name)

@register_trainer('base_trainer')
class Trainer:
    def __init__(self, model, dm, config={}):
        self.model = model
        self.dm = dm
        self.config = config
    
    def setup(self):
        # 数据加载
        self.train_loader, self.val_loader, self.test_loader = self.dm.get_dataloaders()

        # 模型 训练状态加载
        resume_config = self.config.get("resume_config", {})
        resume = resume_config.get("resume", False)
        checkpoint_dir = resume_config.get("resume_from_checkpointDir", None)
        if resume and os.path.isdir(checkpoint_dir):
            checkpoint_epoch = resume_config.get("resume_from_checkpointEpoch", 0)
            epoch, global_step, optimizer_state, scheduler_state, best_val_loss = self.load_fromCheckpoint(
                epoch=checkpoint_epoch,
                checkpoint_dir=checkpoint_dir)
        else:
            epoch, global_step, optimizer_state, scheduler_state, best_val_loss = 0, 0, None, None, float("inf")
        self.optimizer, self.scheduler, self.loss_fn = self.build_train_components(
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state)

        # 训练超参数设置
        self.start_epoch = epoch
        self.max_epochs = self.config.get("max_epochs")
        self.best_val_loss = best_val_loss
        self.normalize = self.config.get("normalize")
        self.device = self.config.get("device")
        self.model.to(self.device)

        # 实验记录设置
        experiment_config = self.config.get("experiment_config", {})
        experiments_dir = experiment_config.get("experiment_dir", "./experiments")
        experiment_name = experiment_config.get("experiment_name", "default_experiment")
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # 实验日志
        log_config = experiment_config.get("log_config", {})
        log_dir = os.path.join(experiment_dir, log_config.get("log_dir", "logs"))
        os.makedirs(log_dir, exist_ok=True)
        self.log_step_interval = log_config.get("log_step_interval", 100)
        self.test_epoch_interval = log_config.get("test_epoch_interval", 20)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = global_step
        self.test_metrics = experiment_config.get("test_config", {}).get("test_metrics", [])
        self.verbose = experiment_config.get("verbose")

        # 检查点
        checkpoint_config = experiment_config.get("checkpoint_config", {})
        self.checkpoint_dir = os.path.join(experiment_dir, checkpoint_config.get("checkpoint_dir", "checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_epoch_interval = checkpoint_config.get("save_epoch_interval", 10)

        # hydra配置保存
        hydra_config = experiment_config.get("hydra_config", {})
        origin_dir = hydra_config.get("origin_dir", None)
        if origin_dir is not None:
            hydra_save_dir = os.path.join(experiment_dir, "hydra")
            os.makedirs(hydra_save_dir, exist_ok=True)
            for item in os.listdir(origin_dir):
                s = os.path.join(origin_dir, item)
                d = os.path.join(hydra_save_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        
        # 初始化指标工具
        data_range = 1.0 if self.normalize else 255.0
        self.psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)


    def train_step(self, batch):
        input, target = batch
        intput = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in input.items()}
        target = target.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(intput)
        loss = self.loss_fn(outputs, target)
        loss.backward()
        self.optimizer.step()

        if (self.global_step + 1) % self.log_step_interval == 0:
            self.writer.add_scalar('Loss/Step', loss, self.global_step + 1)
            lr_img = vutils.make_grid(intput['lr'][:4], normalize=True, scale_each=True)
            hr_img = vutils.make_grid(target[:4], normalize=True, scale_each=True)
            pred_img = vutils.make_grid(outputs[:4], normalize=True, scale_each=True)
            self.writer.add_image('Images/LR', lr_img, self.global_step + 1)
            self.writer.add_image('Images/Pred', pred_img, self.global_step + 1)
            self.writer.add_image('Images/HR', hr_img, self.global_step + 1)
        self.global_step += 1

        return loss.item()
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch}/{self.max_epochs}", ascii=True, disable=not self.verbose)):
            loss = self.train_step(batch)
            epoch_loss += loss
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def val_step(self, batch):
        input, target = batch
        intput = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in input.items()}
        target = target.to(self.device)

        with torch.no_grad():
            outputs = self.model(intput)
            loss = self.loss_fn(outputs, target)
        return loss.item()
    
    def val_epoch(self):
        self.model.eval()

        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(self.val_loader, desc="Validation", ascii=True, disable=not self.verbose)):
            loss = self.val_step(batch)
            epoch_loss += loss
        avg_loss = epoch_loss / len(self.val_loader)
        return avg_loss
    
    def test_step(self, batch):
        input, target = batch
        intput = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in input.items()}
        target = target.to(self.device)

        with torch.no_grad():
            outputs = self.model(intput)

        return outputs, target
    
    def test_epoch(self):
        self.model.eval()

        for batch in tqdm(self.test_loader, desc="Testing", ascii=True, disable=not self.verbose):
            outputs, targets = self.test_step(batch)

            self.psnr_metric.update(outputs, targets)
            self.ssim_metric.update(outputs, targets)

        results = {}
        if "psnr" in self.test_metrics:
            results["PSNR"] = self.psnr_metric.compute().item()
            self.psnr_metric.reset()
        if "ssim" in self.test_metrics:
            results["SSIM"] = self.ssim_metric.compute().item()
            self.ssim_metric.reset()

        return results
    
    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch()
            self.writer.add_scalars('Loss/Epoch', {
                'Train': train_loss,
                'Val': val_loss
            }, epoch+1)

            # 调度器步进
            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.save_checkpoint("best", self.global_step, self.optimizer.state_dict(), self.scheduler.state_dict(), val_loss, self.checkpoint_dir)
                self.best_val_loss = val_loss
                if self.verbose:
                    print(f"Saved checkpoint at epoch best")

            if (epoch + 1) % self.save_epoch_interval == 0:
                self.save_checkpoint(epoch + 1, self.global_step, self.optimizer.state_dict(), self.scheduler.state_dict(), val_loss, self.checkpoint_dir)
                if self.verbose:
                    print(f"Saved checkpoint at epoch {epoch + 1}")
            
            # 定期测试
            if self.test_epoch_interval > 0 and (epoch + 1) % self.test_epoch_interval == 0:
                test_results = self.test_epoch()
                for metric, value in test_results.items():
                    self.writer.add_scalar(f'Test/{metric}', value, epoch+1)

    def save_checkpoint(self, epoch, global_step, optimizer_state, scheduler_state, val_loss, checkpoint_dir):
        # 保存模型权重
        weight_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        self.model.save_weights(weight_path)

        # 保存训练状态
        state_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_trainer_state.pth")
        torch.save(
            {
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer_state,
                'scheduler_state_dict': scheduler_state,
                'val_loss': val_loss
            },
            state_path
        )

    def load_fromCheckpoint(self, epoch, checkpoint_dir):
        # 加载模型权重
        weight_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        self.model.load_weights(weight_path)
        print(f"Loaded model weights from {weight_path}")

        # 加载训练状态
        state_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_trainer_state.pth")
        if not os.path.isfile(state_path):
            print(f"[Warning] No trainer state found at {state_path}")
            return 0, 0, None, None, float('inf')

        trainer_state = torch.load(state_path, map_location="cpu")

        epoch = trainer_state.get("epoch", 0)
        global_step = trainer_state.get("global_step", 0)
        optimizer_state = trainer_state.get("optimizer_state_dict", None)
        scheduler_state = trainer_state.get("scheduler_state_dict", None)
        val_loss = trainer_state.get("val_loss", float("inf"))

        return epoch, global_step, optimizer_state, scheduler_state, val_loss

    def build_train_components(self, optimizer_state=None, scheduler_state=None):
        # optimizer scheduler loss_fn
        component_config = self.config.get("component_config")
        optimizer_config = component_config.get("optimizer_config")
        scheduler_config = component_config.get("scheduler_config")
        loss_config = component_config.get("loss_config")
        optimizer, scheduler, loss_fn = get_train_components(
            model_parameters=self.model.parameters(),
            optimizer_config=optimizer_config,optimizer_state=optimizer_state,
            scheduler_config=scheduler_config,scheduler_state=scheduler_state,
            loss_config=loss_config
        )

        return optimizer, scheduler, loss_fn
