# src/trainers/trainer.py

import os
import torch
from torch.utils.tensorboard import SummaryWriter


default_config = {}
class Trainer:
    def __init__(self, model, dm, config=default_config):
        self.model = model
        self.dm = dm
        self.config = config
    
    def setup(self):
        # 数据加载
        self.train_loader, self.val_loader, self.test_loader = self.dm.get_dataloaders()

        # 模型 训练状态加载
        resume = self.config.get("resume", False)
        checkpoint_dir = self.config.get("checkpoint_dir", "")
        if resume and os.path.isdir(checkpoint_dir):
            checkpoint_epoch = self.config.get("checkpoint_epoch", 0)
            epoch, global_step, optimizer_state, scheduler_state, best_val_loss = self.load_fromCheckpoint(
                epoch=checkpoint_epoch,
                checkpoint_dir=checkpoint_dir)
        else:
            epoch, global_step, optimizer_state, scheduler_state, best_val_loss = 0, 0, None, None, float("inf")
        # 优化器 调度器 损失函数构建
        self.optimizer, self.scheduler = self.build_optimizer_scheduler(
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state)
        self.loss_fn = self.build_loss_function()

        # 训练超参数设置
        self.device = self.config.get("device", "cpu")
        self.start_epoch = epoch
        self.max_epochs = self.config.get("max_epochs", 1000)
        self.best_val_loss = best_val_loss

        # 实验记录设置
        self.experiments_dir = self.config.get("experiment_dir", "./experiments")
        self.experiment_name = self.config.get("experiment_name", "default_experiment")
        self.experiment_dir = os.path.join(self.experiments_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_step_interval = self.config.get("log_step", 100)
        self.save_epoch_interval = self.config.get("save_epoch_interval", 10)
        self.test_epoch_interval = self.config.get("test_epoch_interval", 20)
        self.global_step = global_step
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.test_metrics = self.config.get("test_metrics", [])

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
    
    def test_epoch(self):
        self.model.eval()
        results = {}
        for metric in self.test_metrics:
            results[metric.__name__] = 0.0

        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                for metric in self.test_metrics:
                    results[metric.__name__] += metric(outputs, targets).item()

        for metric_name in results:
            results[metric_name] /= len(self.test_loader)

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

    def build_optimizer_scheduler(self, optimizer_state=None, scheduler_state=None):
        # optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-3)
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get("step_size", 10),
            gamma=self.config.get("gamma", 0.1)
        )
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        return optimizer, scheduler
    
    def build_loss_function(self):
        loss_fn = torch.nn.L1Loss()
        return loss_fn