"""
Evaluation metrics for MILK10k
"""
import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    multilabel_confusion_matrix
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import DIAGNOSIS_CATEGORIES


def compute_metrics(predictions, targets, threshold=0.5):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    targets = targets.astype(np.int32)
    pred_binary = (predictions >= threshold).astype(np.int32)
    
    metrics = {}
    
    metrics['macro_f1'] = f1_score(targets, pred_binary, average='macro', zero_division=0)
    metrics['micro_f1'] = f1_score(targets, pred_binary, average='micro', zero_division=0)
    metrics['weighted_f1'] = f1_score(targets, pred_binary, average='weighted', zero_division=0)
    
    per_class_f1 = f1_score(targets, pred_binary, average=None, zero_division=0)
    metrics['per_class_f1'] = per_class_f1
    for i, category in enumerate(DIAGNOSIS_CATEGORIES):
        metrics[f'f1_{category}'] = per_class_f1[i]
    
    metrics['macro_precision'] = precision_score(targets, pred_binary, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(targets, pred_binary, average='macro', zero_division=0)
    
    try:
        metrics['macro_auc_roc'] = roc_auc_score(targets, predictions, average='macro')
        metrics['micro_auc_roc'] = roc_auc_score(targets, predictions, average='micro')
        
        per_class_auc = roc_auc_score(targets, predictions, average=None)
        for i, category in enumerate(DIAGNOSIS_CATEGORIES):
            metrics[f'auc_{category}'] = per_class_auc[i]
    except ValueError:
        pass
    
    try:
        metrics['macro_ap'] = average_precision_score(targets, predictions, average='macro')
    except ValueError:
        pass
    
    return metrics


def compute_confusion_matrix(predictions, targets, threshold=0.5):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    pred_binary = (predictions >= threshold).astype(int)
    
    cm = multilabel_confusion_matrix(targets, pred_binary)
    
    return cm


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
    
    def compute(self, threshold=0.5):
        if not self.predictions:
            return {}
        
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        predictions = torch.sigmoid(predictions)
        
        metrics = compute_metrics(predictions, targets, threshold)
        
        return metrics


if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics computation...")
    
    # Create dummy data
    num_samples = 100
    num_classes = 11
    
    np.random.seed(42)
    targets = np.random.randint(0, 2, size=(num_samples, num_classes))
    predictions = np.random.rand(num_samples, num_classes)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    print("\nComputed Metrics:")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    
    if 'macro_auc_roc' in metrics:
        print(f"Macro AUC-ROC: {metrics['macro_auc_roc']:.4f}")
    
    print("\nPer-class F1 scores:")
    for category in DIAGNOSIS_CATEGORIES:
        if f'f1_{category}' in metrics:
            print(f"  {category}: {metrics[f'f1_{category}']:.4f}")
    
    # Test confusion matrix
    cm = compute_confusion_matrix(predictions, targets)
    print(f"\nConfusion matrix shape: {cm.shape}")
    
    # Test MetricsTracker
    print("\nTesting MetricsTracker...")
    tracker = MetricsTracker()
    
    for i in range(5):
        batch_preds = torch.rand(20, 11)
        batch_targets = torch.randint(0, 2, (20, 11)).float()
        tracker.update(batch_preds, batch_targets)
    
    metrics = tracker.compute()
    print(f"Tracked Macro F1: {metrics['macro_f1']:.4f}")
    
    print("\nMetrics computation test completed!")
