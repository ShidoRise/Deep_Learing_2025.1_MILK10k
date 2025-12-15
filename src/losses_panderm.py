"""
Compound Loss Functions for PanDerm Model
Includes Soft F1 Loss + Focal Loss combination optimized for Macro F1 metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Union


class SoftF1Loss(nn.Module):
    """Differentiable approximation of F1 score for direct metric optimization."""
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        soft_tp = (probs * targets).sum(dim=0)
        soft_fp = (probs * (1 - targets)).sum(dim=0)
        soft_fn = ((1 - probs) * targets).sum(dim=0)
        
        soft_f1 = (2 * soft_tp) / (2 * soft_tp + soft_fp + soft_fn + self.epsilon)
        macro_f1 = soft_f1.mean()
        
        return 1 - macro_f1


class WeightedFocalLoss(nn.Module):
    """Focal Loss with per-class weights for handling class imbalance."""
    
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = focal_weight * bce
        
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = loss * weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CompoundLoss(nn.Module):
    """Combined Focal + Soft F1 Loss for balanced optimization."""
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.5,
        soft_f1_weight: float = 0.5
    ):
        super().__init__()
        self.focal = WeightedFocalLoss(gamma=focal_gamma, class_weights=class_weights)
        self.soft_f1 = SoftF1Loss()
        self.focal_weight = focal_weight
        self.soft_f1_weight = soft_f1_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        focal_loss = self.focal(logits, targets)
        soft_f1_loss = self.soft_f1(logits, targets)
        total = self.focal_weight * focal_loss + self.soft_f1_weight * soft_f1_loss
        
        return {
            'total': total,
            'focal': focal_loss,
            'soft_f1': soft_f1_loss
        }


class AuxiliaryCompoundLoss(nn.Module):
    """Compound loss with auxiliary heads for deep supervision."""
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.5,
        soft_f1_weight: float = 0.5,
        aux_weight: float = 0.3
    ):
        super().__init__()
        self.main_criterion = CompoundLoss(
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            focal_weight=focal_weight,
            soft_f1_weight=soft_f1_weight
        )
        self.aux_criterion = WeightedFocalLoss(gamma=focal_gamma, class_weights=class_weights)
        self.aux_weight = aux_weight
    
    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if isinstance(outputs, dict):
            main_loss = self.main_criterion(outputs['main'], targets)
            losses = {
                'main': main_loss['total'],
                'focal': main_loss['focal'],
                'soft_f1': main_loss['soft_f1']
            }
            
            total = main_loss['total']
            
            if 'aux_clinical' in outputs:
                aux_clin = self.aux_criterion(outputs['aux_clinical'], targets)
                losses['aux_clinical'] = aux_clin
                total = total + self.aux_weight * aux_clin
            
            if 'aux_dermoscopic' in outputs:
                aux_derm = self.aux_criterion(outputs['aux_dermoscopic'], targets)
                losses['aux_dermoscopic'] = aux_derm
                total = total + self.aux_weight * aux_derm
            
            losses['total'] = total
            return losses
        else:
            return self.main_criterion(outputs, targets)


class ClassBalancedCompoundLoss(nn.Module):
    """Compound loss with class-balanced weighting based on effective number of samples."""
    
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        soft_f1_weight: float = 0.5,
        aux_weight: float = 0.3
    ):
        super().__init__()
        
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        self.criterion = AuxiliaryCompoundLoss(
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            focal_weight=focal_weight,
            soft_f1_weight=soft_f1_weight,
            aux_weight=aux_weight
        )
    
    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.criterion(outputs, targets)


def create_panderm_loss(
    samples_per_class: Optional[List[int]] = None,
    focal_gamma: float = 2.0,
    focal_weight: float = 0.5,
    soft_f1_weight: float = 0.5,
    aux_weight: float = 0.3,
    use_class_balance: bool = True
) -> nn.Module:
    """Factory function to create loss for PanDerm model."""
    if use_class_balance and samples_per_class is not None:
        return ClassBalancedCompoundLoss(
            samples_per_class=samples_per_class,
            focal_gamma=focal_gamma,
            focal_weight=focal_weight,
            soft_f1_weight=soft_f1_weight,
            aux_weight=aux_weight
        )
    else:
        return AuxiliaryCompoundLoss(
            focal_gamma=focal_gamma,
            focal_weight=focal_weight,
            soft_f1_weight=soft_f1_weight,
            aux_weight=aux_weight
        )


if __name__ == "__main__":
    print("Testing PanDerm Loss Functions")
    print("=" * 50)
    
    batch_size = 8
    num_classes = 11
    samples_per_class = [1000, 800, 500, 400, 300, 200, 150, 100, 80, 50, 30]
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print(f"\nInput: logits={logits.shape}, targets={targets.shape}")
    
    # Test SoftF1Loss
    soft_f1 = SoftF1Loss()
    loss = soft_f1(logits, targets)
    print(f"\n1. Soft F1 Loss: {loss.item():.4f}")
    
    # Test CompoundLoss
    compound = CompoundLoss(focal_gamma=2.0)
    losses = compound(logits, targets)
    print(f"\n2. Compound Loss:")
    for k, v in losses.items():
        print(f"   {k}: {v.item():.4f}")
    
    # Test with auxiliary heads
    outputs = {
        'main': logits,
        'aux_clinical': torch.randn(batch_size, num_classes),
        'aux_dermoscopic': torch.randn(batch_size, num_classes)
    }
    
    aux_compound = AuxiliaryCompoundLoss(aux_weight=0.3)
    losses = aux_compound(outputs, targets)
    print(f"\n3. Auxiliary Compound Loss:")
    for k, v in losses.items():
        print(f"   {k}: {v.item():.4f}")
    
    # Test class-balanced version
    cb_loss = ClassBalancedCompoundLoss(samples_per_class=samples_per_class)
    losses = cb_loss(outputs, targets)
    print(f"\n4. Class-Balanced Compound Loss:")
    for k, v in losses.items():
        print(f"   {k}: {v.item():.4f}")
    
    # Test factory function
    loss_fn = create_panderm_loss(samples_per_class=samples_per_class)
    losses = loss_fn(outputs, targets)
    print(f"\n5. Factory Created Loss:")
    print(f"   total: {losses['total'].item():.4f}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")

