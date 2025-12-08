"""
Training script for EfficientNetV2-L skin lesion classification
Optimized for A100 80GB GPU with enhanced loss functions and deep supervision
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
import copy
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from utils import (
    set_seed, 
    get_device, 
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    EarlyStopping,
    format_time
)
from dataset import MILK10kDataset, get_transforms, get_dataloaders
from models_v2 import create_model_v2, SkinLesionClassifierV2
from losses import (
    CombinedLoss, 
    AsymmetricLoss, 
    ClassBalancedFocalLoss,
    FocalLoss,
    get_samples_per_class
)
from evaluate import compute_metrics, MetricsTracker


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    """
    
    def __init__(
        self,
        optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-7,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr + (self.max_lr - base_lr) * 
                (1 + np.cos(np.pi * (self.step_in_cycle - self.warmup_steps) / 
                           (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), 
                                   self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = np.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class TrainerV2:
    """
    Enhanced trainer for EfficientNetV2-L with:
    - Deep supervision (auxiliary heads)
    - EMA (Exponential Moving Average)
    - Combined loss function
    - Gradient accumulation
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader, 
        val_loader, 
        class_weights=None,
        samples_per_class=None,
        device='cuda',
        config=None
    ):
        self.config = config or TRAIN_CONFIG_V2
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Estimate samples per class if not provided
        if samples_per_class is None and class_weights is not None:
            total_samples = len(train_loader.dataset)
            samples_per_class = get_samples_per_class(class_weights, total_samples)
        
        self.samples_per_class = samples_per_class
        
        # Setup loss function
        if samples_per_class is not None:
            self.criterion = CombinedLoss(
                samples_per_class=samples_per_class,
                loss_type=LOSS_CONFIG_V2.get('type', 'asymmetric'),
                use_label_smoothing=LOSS_CONFIG_V2.get('use_label_smoothing', True),
                aux_weight=self.config.get('aux_loss_weight', 0.3),
                gamma=LOSS_CONFIG_V2.get('focal_gamma', 2.0)
            )
        else:
            # Fallback to simple focal loss
            alpha = None
            if class_weights is not None:
                alpha = torch.tensor([class_weights[cat] for cat in DIAGNOSIS_CATEGORIES]).to(device)
            self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config['num_epochs']
        warmup_steps = steps_per_epoch * self.config.get('warmup_epochs', 3)
        
        if self.config.get('scheduler') == 'cosine_warmup':
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=total_steps,
                max_lr=self.config['learning_rate'],
                min_lr=self.config['min_lr'],
                warmup_steps=warmup_steps
            )
            self.scheduler_per_step = True
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config['min_lr']
            )
            self.scheduler_per_step = False
        
        # Mixed precision
        self.use_amp = self.config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA
        self.use_ema = self.config.get('use_ema', True)
        if self.use_ema:
            self.ema = EMA(model, decay=self.config.get('ema_decay', 0.9999))
        
        # Gradient accumulation
        self.grad_accum_steps = self.config.get('gradient_accumulation', 1)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 12),
            mode='max',
            verbose=True
        )
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_macro': [],
            'val_f1_micro': [],
            'learning_rate': []
        }
        
        # TensorBoard
        log_dir = Path(self.config['log_dir']) / f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}")
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        
        loss_meter = AverageMeter()
        main_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()
        
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]'
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Parse batch
            if len(batch) == 3:
                images, labels, metadata = batch
                labels = labels.to(self.device)
                metadata = metadata.to(self.device)
            else:
                images, labels = batch
                labels = labels.to(self.device)
                metadata = None
            
            # Handle image tuple (clinical, dermoscopic)
            if isinstance(images, (tuple, list)):
                images = (images[0].to(self.device), images[1].to(self.device))
            else:
                images = images.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images, metadata)
                    
                    # Compute loss
                    if isinstance(outputs, dict):
                        losses = self.criterion(outputs, labels)
                        loss = losses['total'] / self.grad_accum_steps
                    else:
                        loss = self.criterion({'main': outputs}, labels)['total'] / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    if self.config.get('gradient_clip'):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['gradient_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.use_ema:
                        self.ema.update()
                    
                    # Step scheduler (if per-step)
                    if self.scheduler_per_step:
                        self.scheduler.step()
            else:
                outputs = self.model(images, metadata)
                
                if isinstance(outputs, dict):
                    losses = self.criterion(outputs, labels)
                    loss = losses['total'] / self.grad_accum_steps
                else:
                    loss = self.criterion({'main': outputs}, labels)['total'] / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.config.get('gradient_clip'):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['gradient_clip']
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.use_ema:
                        self.ema.update()
                    
                    if self.scheduler_per_step:
                        self.scheduler.step()
            
            # Update meters
            loss_value = loss.item() * self.grad_accum_steps
            loss_meter.update(loss_value, labels.size(0))
            
            if isinstance(outputs, dict):
                main_loss_meter.update(losses['main'].item(), labels.size(0))
                if 'aux_clinical' in losses:
                    aux_loss_meter.update(
                        (losses['aux_clinical'].item() + losses.get('aux_dermoscopic', losses['aux_clinical']).item()) / 2,
                        labels.size(0)
                    )
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'main': f'{main_loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return loss_meter.avg
    
    def validate_epoch(self, epoch: int, use_ema: bool = True):
        # Apply EMA weights for validation
        if self.use_ema and use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_outputs = []
        all_labels = []
        
        pbar = tqdm(
            self.val_loader, 
            desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Val]'
        )
        
        with torch.no_grad():
            for batch in pbar:
                if len(batch) == 3:
                    images, labels, metadata = batch
                    labels = labels.to(self.device)
                    metadata = metadata.to(self.device)
                else:
                    images, labels = batch
                    labels = labels.to(self.device)
                    metadata = None
                
                if isinstance(images, (tuple, list)):
                    images = (images[0].to(self.device), images[1].to(self.device))
                else:
                    images = images.to(self.device)
                
                # Forward pass (eval mode returns single tensor)
                outputs = self.model(images, metadata)
                
                # Handle both dict and tensor outputs
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs
                
                # Simple loss for validation
                loss = FocalLoss(gamma=2.0)(logits, labels)
                
                loss_meter.update(loss.item(), labels.size(0))
                
                all_outputs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Restore original weights
        if self.use_ema and use_ema:
            self.ema.restore()
        
        # Compute metrics
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels).astype(np.int32)
        
        metrics = compute_metrics(all_outputs, all_labels, threshold=0.5)
        
        return loss_meter.avg, metrics
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            start_epoch: Epoch to resume from
            best_f1: Best F1 score from previous training
        """
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Model weights loaded")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Optimizer state loaded")
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Scheduler state loaded")
        
        # Load EMA if available
        if self.use_ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
            print(f"  EMA state loaded")
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('metrics', 0.0)
        if isinstance(best_f1, dict):
            best_f1 = best_f1.get('macro_f1', 0.0)
        
        print(f"  Resuming from epoch {start_epoch + 1}")
        print(f"  Previous best F1: {best_f1:.4f}")
        
        return start_epoch, best_f1
    
    def train(self, start_epoch: int = 0, best_f1: float = 0.0):
        print("\n" + "=" * 60)
        print("STARTING TRAINING - EfficientNetV2-L")
        print("=" * 60)
        print(f"Device: {self.device}")
        total_params, trainable_params = count_parameters(self.model)
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.config['batch_size'] * self.grad_accum_steps}")
        print(f"Number of epochs: {self.config['num_epochs']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Use EMA: {self.use_ema}")
        print(f"Mixed precision: {self.use_amp}")
        if start_epoch > 0:
            print(f"Resuming from epoch: {start_epoch + 1}")
            print(f"Previous best F1: {best_f1:.4f}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Step scheduler (if per-epoch)
            if not self.scheduler_per_step:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(val_metrics['macro_f1'])
            self.history['val_f1_micro'].append(val_metrics['micro_f1'])
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('F1/macro', val_metrics['macro_f1'], epoch)
            self.writer.add_scalar('F1/micro', val_metrics['micro_f1'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for i, category in enumerate(DIAGNOSIS_CATEGORIES):
                self.writer.add_scalar(
                    f'F1_per_class/{category}', 
                    val_metrics['per_class_f1'][i], 
                    epoch
                )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - {format_time(epoch_time)}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['macro_f1']:.4f}")
            print(f"  Val F1 (Micro): {val_metrics['micro_f1']:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                
                # Save EMA weights if using EMA
                if self.use_ema:
                    self.ema.apply_shadow()
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['macro_f1'],
                    self.config['checkpoint_dir'],
                    filename='best_model_v2.pth',
                    scheduler=self.scheduler
                )
                
                if self.use_ema:
                    self.ema.restore()
                
                print(f"  ✅ Best model saved! F1: {best_f1:.4f}")
            
            # Save periodic checkpoints (with all states for resume)
            if (epoch + 1) % self.config['save_every'] == 0:
                # Save full checkpoint with EMA state for resume
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'metrics': val_metrics['macro_f1'],
                    'ema_shadow': self.ema.shadow if self.use_ema else None,
                    'config': self.config,
                    'timestamp': datetime.now().isoformat()
                }
                checkpoint_path = Path(self.config['checkpoint_dir']) / f'checkpoint_v2_epoch_{epoch+1}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"  📁 Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                break
            
            print("-" * 60)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best F1 Score (Macro): {best_f1:.4f}")
        print(f"Best model saved: {self.config['checkpoint_dir']}/best_model_v2.pth")
        print("=" * 60 + "\n")
        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_path = Path(self.config['checkpoint_dir']) / 'training_history_v2.csv'
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved: {history_path}")
        
        self.writer.close()
        
        return self.history


def main():
    """Main training function for local execution"""
    set_seed(TRAIN_CONFIG_V2['random_seed'])
    
    Path(TRAIN_CONFIG_V2['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_CONFIG_V2['log_dir']).mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    # Load data
    print("Loading preprocessed data...")
    train_df = pd.read_csv(PREPROCESSED_DIR / 'train_data.csv')
    val_df = pd.read_csv(PREPROCESSED_DIR / 'val_data.csv')
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    
    with open(PREPROCESSED_DIR / 'class_weights.json', 'r') as f:
        class_weights = json.load(f)
    
    # Estimate samples per class
    total_samples = len(train_df)
    samples_per_class = get_samples_per_class(class_weights, total_samples)
    print(f"\nEstimated samples per class: {samples_per_class}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        train_df,
        val_df,
        batch_size=TRAIN_CONFIG_V2['batch_size'],
        num_workers=TRAIN_CONFIG_V2['num_workers'],
        image_size=IMAGE_CONFIG_V2['image_size'],
        fusion_strategy='late',  # Always late for V2
        use_metadata=MODEL_CONFIG_V2['use_metadata']
    )
    
    # Create model
    print("\nCreating EfficientNetV2-L model...")
    model = create_model_v2(
        architecture=MODEL_CONFIG_V2['architecture'],
        num_classes=len(DIAGNOSIS_CATEGORIES),
        pretrained=MODEL_CONFIG_V2['pretrained'],
        use_metadata=MODEL_CONFIG_V2['use_metadata'],
        metadata_dim=MODEL_CONFIG_V2['metadata_dim'],
        dropout=MODEL_CONFIG_V2['dropout'],
        use_auxiliary_heads=MODEL_CONFIG_V2['use_auxiliary_heads']
    )
    
    print(f"Model: {MODEL_CONFIG_V2['architecture']}")
    print(f"Image size: {IMAGE_CONFIG_V2['image_size']}")
    print(f"Use metadata: {MODEL_CONFIG_V2['use_metadata']}")
    print(f"Auxiliary heads: {MODEL_CONFIG_V2['use_auxiliary_heads']}")
    count_parameters(model)
    
    # Create trainer
    trainer = TrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        samples_per_class=samples_per_class,
        device=device,
        config=TRAIN_CONFIG_V2
    )
    
    # Train
    history = trainer.train()
    
    print("\n✅ Training script completed successfully!")


if __name__ == "__main__":
    main()

