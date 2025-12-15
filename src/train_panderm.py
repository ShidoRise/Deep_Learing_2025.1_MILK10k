"""
Training script for PanDerm Tri-Modal Fusion Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime
import sys
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent))

from config import *
from utils import set_seed, get_device, count_parameters, save_checkpoint, AverageMeter, EarlyStopping, format_time
from models_panderm import create_panderm_model, get_layer_wise_lr_params
from losses_panderm import create_panderm_loss
from evaluate import compute_metrics


class PanDermDataset(Dataset):
    """Dataset for PanDerm model with separate MONET scores."""
    
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        
        self.monet_cols = [col for col in df.columns if 'MONET_' in col]
        self.clinical_cols = ['age_approx', 'sex', 'skin_tone_class', 'site']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        clinical_img = self._load_image(row['clinical_image_path'])
        dermoscopic_img = self._load_image(row['dermoscopic_image_path'])
        
        if self.transform:
            clinical_img = self.transform(image=clinical_img)['image']
            dermoscopic_img = self.transform(image=dermoscopic_img)['image']
        
        monet_scores = self._get_monet_scores(row)
        metadata = self._get_metadata(row)
        
        if self.is_test:
            return clinical_img, dermoscopic_img, monet_scores, metadata
        
        labels = torch.tensor(row[DIAGNOSIS_CATEGORIES].values.astype(np.float32))
        return clinical_img, dermoscopic_img, monet_scores, metadata, labels
    
    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _get_monet_scores(self, row):
        scores = []
        for col in self.monet_cols[:7]:
            val = row.get(col, 0.0)
            scores.append(float(val) if not pd.isna(val) else 0.0)
        while len(scores) < 7:
            scores.append(0.0)
        return torch.tensor(scores, dtype=torch.float32)
    
    def _get_metadata(self, row):
        metadata = []
        
        age = row.get('age_approx', 50.0)
        metadata.append(float(age) / 100.0 if not pd.isna(age) else 0.5)
        
        sex_map = {'male': 0.0, 'female': 1.0}
        sex = str(row.get('sex', 'unknown')).lower()
        metadata.append(sex_map.get(sex, 0.5))
        
        skin_tone = row.get('skin_tone_class', row.get('skin_tone', 3.0))
        metadata.append(float(skin_tone) / 5.0 if not pd.isna(skin_tone) else 0.6)
        
        site_map = {'head_neck_face': 0, 'upper_extremity': 1, 'lower_extremity': 2, 
                    'trunk': 3, 'palms_soles': 4, 'oral_genital': 5}
        site = str(row.get('site', row.get('anatom_site_general', 'unknown'))).lower()
        site_val = site_map.get(site, 3) / 5.0
        metadata.append(site_val)
        
        for col in self.monet_cols[:7]:
            val = row.get(col, 0.0)
            metadata.append(float(val) if not pd.isna(val) else 0.0)
        
        while len(metadata) < 11:
            metadata.append(0.0)
        
        return torch.tensor(metadata[:11], dtype=torch.float32)


def get_panderm_transforms(image_size=224, augment=True):
    mean = IMAGE_CONFIG_PANDERM['normalize_mean']
    std = IMAGE_CONFIG_PANDERM['normalize_std']
    
    if augment:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=image_size//8, max_width=image_size//8, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    return transform


def get_panderm_dataloaders(train_df, val_df, batch_size=32, num_workers=4, image_size=224):
    train_transform = get_panderm_transforms(image_size, augment=True)
    val_transform = get_panderm_transforms(image_size, augment=False)
    
    train_dataset = PanDermDataset(train_df, transform=train_transform)
    val_dataset = PanDermDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def apply_dropout_augmentation(clinical_img, monet_scores, modality_drop=0.2, concept_drop=0.1, training=True):
    if not training:
        return clinical_img, monet_scores
    
    B = clinical_img.shape[0]
    device = clinical_img.device
    
    if modality_drop > 0:
        mask = (torch.rand(B, 1, 1, 1, device=device) > modality_drop).float()
        clinical_img = clinical_img * mask
    
    if concept_drop > 0:
        mask = (torch.rand_like(monet_scores) > concept_drop).float()
        monet_scores = monet_scores * mask
    
    return clinical_img, monet_scores


class PanDermTrainer:
    def __init__(self, model, train_loader, val_loader, samples_per_class=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = TRAIN_CONFIG_PANDERM
        
        self.criterion = create_panderm_loss(
            samples_per_class=samples_per_class,
            focal_gamma=LOSS_CONFIG_PANDERM['focal_gamma'],
            focal_weight=LOSS_CONFIG_PANDERM['focal_weight'],
            soft_f1_weight=LOSS_CONFIG_PANDERM['soft_f1_weight'],
            aux_weight=LOSS_CONFIG_PANDERM['aux_loss_weight']
        )
        
        param_groups = get_layer_wise_lr_params(
            model, 
            base_lr=self.config['base_lr'],
            decay_rate=self.config['backbone_lr_decay']
        )
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.config['weight_decay'])
        
        total_steps = len(train_loader) * self.config['num_epochs']
        warmup_steps = len(train_loader) * self.config['warmup_epochs']
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[g['lr'] for g in param_groups],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            final_div_factor=self.config['base_lr'] / self.config['min_lr']
        )
        
        self.scaler = torch.amp.GradScaler('cuda')
        self.early_stopping = EarlyStopping(patience=self.config['early_stopping_patience'], mode='max')
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'learning_rate': []}
        
        log_dir = Path(self.config['log_dir']) / f"panderm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
    
    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        accum_steps = self.config.get('gradient_accumulation', 1)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            clinical_img, dermoscopic_img, monet_scores, metadata, labels = batch
            clinical_img = clinical_img.to(self.device)
            dermoscopic_img = dermoscopic_img.to(self.device)
            monet_scores = monet_scores.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)
            
            clinical_img, monet_scores = apply_dropout_augmentation(
                clinical_img, monet_scores,
                modality_drop=self.config.get('modality_dropout', 0.2),
                concept_drop=self.config.get('concept_dropout', 0.1),
                training=True
            )
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(clinical_img, dermoscopic_img, monet_scores, metadata)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict['total'] / accum_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            loss_meter.update(loss_dict['total'].item(), labels.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
        
        return loss_meter.avg
    
    def validate_epoch(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        all_outputs, all_labels = [], []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                clinical_img, dermoscopic_img, monet_scores, metadata, labels = batch
                clinical_img = clinical_img.to(self.device)
                dermoscopic_img = dermoscopic_img.to(self.device)
                monet_scores = monet_scores.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(clinical_img, dermoscopic_img, monet_scores, metadata)
                
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs
                
                loss_dict = self.criterion(outputs, labels)
                loss_meter.update(loss_dict['total'].item(), labels.size(0))
                
                all_outputs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels).astype(np.int32)
        metrics = compute_metrics(all_outputs, all_labels, threshold=0.5)
        
        return loss_meter.avg, metrics
    
    def train(self):
        print("\n" + "=" * 60)
        print("PANDERM TRAINING")
        print("=" * 60)
        count_parameters(self.model)
        
        best_f1 = 0.0
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            current_lr = self.scheduler.get_last_lr()[0]
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(val_metrics['macro_f1'])
            self.history['learning_rate'].append(current_lr)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('F1/macro', val_metrics['macro_f1'], epoch)
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - {format_time(epoch_time)}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['macro_f1']:.4f}, LR: {current_lr:.2e}")
            
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                save_checkpoint(self.model, self.optimizer, epoch, best_f1, 
                               self.config['checkpoint_dir'], filename='panderm_best.pth')
                print(f"  Best model saved! F1: {best_f1:.4f}")
            
            # Save checkpoint every 5 epochs (keep only 1 to save space)
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(self.config['checkpoint_dir']) / 'panderm_checkpoint.pth'
                # Delete old checkpoint if exists
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"  Deleted old checkpoint")
                # Save new checkpoint
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics['macro_f1'],
                               self.config['checkpoint_dir'], filename='panderm_checkpoint.pth')
                print(f"  Checkpoint saved at epoch {epoch+1}")
            
            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best F1: {best_f1:.4f}")
        
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(Path(self.config['checkpoint_dir']) / 'panderm_history.csv', index=False)
        self.writer.close()
        
        return self.history


def get_samples_per_class(train_df):
    samples = []
    for cat in DIAGNOSIS_CATEGORIES:
        if cat in train_df.columns:
            samples.append(int(train_df[cat].sum()))
        else:
            samples.append(100)
    return samples


def main():
    set_seed(TRAIN_CONFIG_PANDERM['random_seed'])
    Path(TRAIN_CONFIG_PANDERM['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_CONFIG_PANDERM['log_dir']).mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    print("Loading data...")
    train_df = pd.read_csv(PREPROCESSED_DIR / 'train_data.csv')
    val_df = pd.read_csv(PREPROCESSED_DIR / 'val_data.csv')
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    samples_per_class = get_samples_per_class(train_df)
    print(f"Samples per class: {samples_per_class}")
    
    train_loader, val_loader = get_panderm_dataloaders(
        train_df, val_df,
        batch_size=TRAIN_CONFIG_PANDERM['batch_size'],
        num_workers=TRAIN_CONFIG_PANDERM['num_workers'],
        image_size=IMAGE_CONFIG_PANDERM['image_size']
    )
    
    print("Creating PanDerm model...")
    model = create_panderm_model(pretrained=True)
    
    trainer = PanDermTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        samples_per_class=samples_per_class,
        device=device
    )
    
    trainer.train()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

