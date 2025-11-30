"""
Training script for MILK10k skin lesion classification
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
from models import create_model
from evaluate import compute_metrics, MetricsTracker


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            F_loss = self.alpha * F_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class Trainer:
    """Trainer class for model training and validation"""
    
    def __init__(self, model, train_loader, val_loader, class_weights=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup loss function
        if LOSS_CONFIG['use_focal_loss']:
            alpha = None
            if class_weights is not None:
                alpha = torch.tensor([class_weights[cat] for cat in DIAGNOSIS_CATEGORIES]).to(device)
            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=LOSS_CONFIG['focal_gamma']
            )
        else:
            if class_weights is not None:
                weight = torch.tensor([class_weights[cat] for cat in DIAGNOSIS_CATEGORIES]).to(device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # Setup scheduler
        if TRAIN_CONFIG['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=TRAIN_CONFIG['num_epochs'],
                eta_min=TRAIN_CONFIG['min_lr']
            )
        elif TRAIN_CONFIG['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.use_amp = TRAIN_CONFIG['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=TRAIN_CONFIG['early_stopping_patience'],
            mode='max',  # Maximize F1 score
            verbose=True
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_macro': [],
            'val_f1_micro': [],
            'learning_rate': []
        }
        
        # TensorBoard writer
        log_dir = Path(TRAIN_CONFIG['log_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 3:
                images, labels, metadata = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                metadata = metadata.to(self.device)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                metadata = None
            
            # Handle different fusion strategies
            if isinstance(images, tuple):
                images = (images[0].to(self.device), images[1].to(self.device))
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    if metadata is not None:
                        outputs = self.model(images, metadata)
                    else:
                        outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if metadata is not None:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), labels.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return loss_meter.avg
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_outputs = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Unpack batch
                if len(batch) == 3:
                    images, labels, metadata = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    metadata = metadata.to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    metadata = None
                
                # Handle different fusion strategies
                if isinstance(images, tuple):
                    images = (images[0].to(self.device), images[1].to(self.device))
                
                # Forward pass
                if metadata is not None:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                loss_meter.update(loss.item(), labels.size(0))
                
                # Store predictions and labels
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Concatenate all predictions
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_outputs, threshold=0.5)
        
        return loss_meter.avg, metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        total_params, trainable_params = count_parameters(self.model)
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
        print(f"Number of epochs: {TRAIN_CONFIG['num_epochs']}")
        print(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")
        print("="*60 + "\n")
        
        best_f1 = 0.0
        start_time = time.time()
        
        for epoch in range(TRAIN_CONFIG['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['macro_f1'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(val_metrics['macro_f1'])
            self.history['val_f1_micro'].append(val_metrics['micro_f1'])
            self.history['learning_rate'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('F1/macro', val_metrics['macro_f1'], epoch)
            self.writer.add_scalar('F1/micro', val_metrics['micro_f1'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log per-class F1 scores
            for i, category in enumerate(DIAGNOSIS_CATEGORIES):
                self.writer.add_scalar(f'F1_per_class/{category}', val_metrics['per_class_f1'][i], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} - {format_time(epoch_time)}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['macro_f1']:.4f}")
            print(f"  Val F1 (Micro): {val_metrics['micro_f1']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['macro_f1'],
                    TRAIN_CONFIG['checkpoint_dir'],
                    filename='best_model.pth'
                )
                print(f"  ✅ Best model saved! F1: {best_f1:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % TRAIN_CONFIG['save_every'] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['macro_f1'],
                    TRAIN_CONFIG['checkpoint_dir'],
                    filename=f'checkpoint_epoch_{epoch+1}.pth'
                )
            
            # Early stopping
            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                break
            
            print("-"*60)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best F1 Score (Macro): {best_f1:.4f}")
        print(f"Best model saved: {TRAIN_CONFIG['checkpoint_dir']}/best_model.pth")
        print("="*60 + "\n")
        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_path = Path(TRAIN_CONFIG['checkpoint_dir']) / 'training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved: {history_path}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history


def main():
    """Main training function"""
    # Set random seed
    set_seed(TRAIN_CONFIG['random_seed'])
    
    # Create necessary directories
    Path(TRAIN_CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv(PREPROCESSED_DIR / 'train_data.csv')
    val_df = pd.read_csv(PREPROCESSED_DIR / 'val_data.csv')
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    
    # Load class weights
    with open(PREPROCESSED_DIR / 'class_weights.json', 'r') as f:
        class_weights = json.load(f)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        train_df,
        val_df,
        batch_size=TRAIN_CONFIG['batch_size'],
        num_workers=TRAIN_CONFIG['num_workers'],
        image_size=IMAGE_CONFIG['image_size'],
        fusion_strategy=IMAGE_CONFIG['fusion_strategy'],
        use_metadata=MODEL_CONFIG['use_metadata']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        architecture=MODEL_CONFIG['architecture'],
        num_classes=len(DIAGNOSIS_CATEGORIES),
        pretrained=MODEL_CONFIG['pretrained'],
        fusion_strategy=IMAGE_CONFIG['fusion_strategy'],
        use_metadata=MODEL_CONFIG['use_metadata'],
        metadata_dim=MODEL_CONFIG['metadata_dim'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    print(f"Model: {MODEL_CONFIG['architecture']}")
    print(f"Fusion strategy: {IMAGE_CONFIG['fusion_strategy']}")
    print(f"Use metadata: {MODEL_CONFIG['use_metadata']}")
    count_parameters(model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device
    )
    
    # Start training
    history = trainer.train()
    
    print("\n✅ Training script completed successfully!")


if __name__ == "__main__":
    main()
