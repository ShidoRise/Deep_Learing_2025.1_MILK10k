"""
Utility functions for MILK10k project
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_device():
    """Get available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, filename='checkpoint.pth', scheduler=None):
    """Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Metrics value (e.g., F1 score)
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename (default: 'checkpoint.pth')
        scheduler: Optional learning rate scheduler
    """
    # Ensure checkpoint directory exists
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = Path(checkpoint_dir) / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer=None, filepath=None, device='cuda', scheduler=None):
    """Load model checkpoint
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state
        filepath: Path to checkpoint file
        device: Device to load checkpoint to
        scheduler: Optional scheduler to load state
    
    Returns:
        tuple: (epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, metrics


def plot_training_history(history, save_path=None):
    """Plot training history (loss and metrics)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot F1 score
    axes[1].plot(history['train_f1'], label='Train F1')
    axes[1].plot(history['val_f1'], label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Training and Validation F1 Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved: {save_path}")
    
    plt.show()


def plot_class_distribution(class_counts, class_names, save_path=None):
    """Plot class distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, class_counts)
    
    # Color bars based on count
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_counts)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Diagnosis Category')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Class Distribution in Training Set')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved: {save_path}")
    
    plt.show()


def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved: {filepath}")


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""
    def __init__(self, patience=10, mode='max', delta=0, verbose=True):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"Validation metric improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
        
        return self.early_stop


def format_time(seconds):
    """Format time in seconds to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
