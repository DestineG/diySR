# src/trainers/trainer.py

import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter

from .components import get_train_components


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
        self.device = self.config.get("device", "cpu")
        self.start_epoch = epoch
        self.max_epochs = self.config.get("max_epochs", 1000)
        self.best_val_loss = best_val_loss

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

    def train_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            self.global_step += 1
            epoch_loss += loss
            if (self.global_step + 1) % self.log_step_interval == 0:
                self.writer.add_scalar('Loss/Step', loss, self.global_step)

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def val_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        return loss.item()
    
    def val_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        for step, batch in enumerate(self.val_loader):
            loss = self.val_step(batch)
            epoch_loss += loss
        avg_loss = epoch_loss / len(self.val_loader)

        return avg_loss
    
    def test_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs, targets
    
    def test_epoch(self):
        self.model.eval()
        all_outputs = []
        all_targets = []
        for step, batch in enumerate(self.test_loader):
            outputs, targets = self.test_step(batch)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        results = {}
        for metric in self.test_metrics:
            if metric == "MAE":
                mae = torch.mean(torch.abs(all_outputs - all_targets)).item()
                results["MAE"] = mae
            elif metric == "MSE":
                mse = torch.mean((all_outputs - all_targets) ** 2).item()
                results["MSE"] = mse
            # 可以添加更多指标计算

        return results
    
    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            self.writer.add_scalars('Loss/Epoch', {
                'Train': train_loss,
                'Val': val_loss
            }, epoch)

            # 调度器步进
            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.save_checkpoint("best", self.global_step, self.optimizer.state_dict(), self.scheduler.state_dict(), val_loss, self.checkpoint_dir)
                self.best_val_loss = val_loss
                print(f"Saved checkpoint at epoch best")

            if (epoch + 1) % self.save_epoch_interval == 0:
                self.save_checkpoint(epoch + 1, self.global_step, self.optimizer.state_dict(), self.scheduler.state_dict(), val_loss, self.checkpoint_dir)
                print(f"Saved checkpoint at epoch {epoch + 1}")
            
            # 定期测试
            if self.test_epoch_interval > 0 and (epoch + 1) % self.test_epoch_interval == 0:
                test_results = self.test_epoch()
                for metric, value in test_results.items():
                    self.writer.add_scalar(f'Test/{metric}', value, epoch)

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
