"""
Enhanced Loss Functions for MILK10k Skin Lesion Classification
Includes class-balanced losses and label smoothing for improved Macro F1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            F_loss = self.alpha * F_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss based on effective number of samples.
    
    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    
    This loss re-weights each class based on the effective number of samples,
    which is more effective than inverse frequency weighting for long-tailed distributions.
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            samples_per_class: List of sample counts for each class
            beta: Hyperparameter for effective number calculation (0.9-0.9999)
            gamma: Focal loss gamma parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        
        # Calculate weights
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B, C) multi-hot labels
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # Focal weighting
        focal_weight = (1 - pt) ** self.gamma
        
        # Class-balanced weighting (move weights to same device as inputs)
        weights = self.weights.to(inputs.device)
        cb_loss = weights.unsqueeze(0) * focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return cb_loss.mean()
        elif self.reduction == 'sum':
            return cb_loss.sum()
        else:
            return cb_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    
    Applies different focusing parameters for positive and negative samples,
    which is particularly effective for multi-label classification with
    class imbalance.
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean',
        disable_torch_grad_focal_loss: bool = False
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (higher = less focus on easy negatives)
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping to prevent training instability
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B, C) multi-hot labels
        """
        # Compute probabilities
        p = torch.sigmoid(inputs)
        
        # Positive and negative probabilities
        p_pos = p
        p_neg = 1 - p
        
        # Asymmetric clipping for negatives
        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
        
        # Basic BCE
        loss_pos = targets * torch.log(p_pos.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=1e-8))
        
        # Asymmetric focusing
        if self.disable_torch_grad_focal_loss:
            p_pos = p_pos.detach()
            p_neg = p_neg.detach()
        
        focal_weight_pos = (1 - p_pos) ** self.gamma_pos
        focal_weight_neg = p_pos ** self.gamma_neg
        
        loss = -(focal_weight_pos * loss_pos + focal_weight_neg * loss_neg)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLabelSmoothing(nn.Module):
    """
    Asymmetric Label Smoothing for Multi-Label Classification.
    
    Applies less smoothing to minority classes to preserve their importance,
    which helps improve Macro F1 score.
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        smoothing_pos: float = 0.1,
        smoothing_neg: float = 0.05,
        adaptive: bool = True
    ):
        """
        Args:
            samples_per_class: List of sample counts for each class
            smoothing_pos: Base smoothing for positive labels (1 -> 1-smoothing)
            smoothing_neg: Base smoothing for negative labels (0 -> smoothing)
            adaptive: Whether to adapt smoothing based on class frequency
        """
        super().__init__()
        self.smoothing_pos = smoothing_pos
        self.smoothing_neg = smoothing_neg
        self.adaptive = adaptive
        
        if adaptive:
            # Less smoothing for minority classes
            total = sum(samples_per_class)
            frequencies = np.array(samples_per_class) / total
            
            # Inverse frequency scaling (minority classes get less smoothing)
            self.register_buffer(
                'smoothing_scale',
                torch.tensor(frequencies / frequencies.max(), dtype=torch.float32)
            )
        else:
            self.register_buffer('smoothing_scale', torch.ones(len(samples_per_class)))
    
    def smooth_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric label smoothing."""
        # Positive smoothing: 1 -> 1 - smoothing_pos * scale
        # Negative smoothing: 0 -> smoothing_neg * scale
        
        # Move smoothing_scale to same device as targets
        smoothing_scale = self.smoothing_scale.to(targets.device)
        
        smoothing_pos = self.smoothing_pos * smoothing_scale.unsqueeze(0)
        smoothing_neg = self.smoothing_neg * smoothing_scale.unsqueeze(0)
        
        smoothed = targets * (1 - smoothing_pos) + (1 - targets) * smoothing_neg
        
        return smoothed


class MultiLabelLDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss adapted for multi-label.
    
    Reference: "Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
    
    Applies class-dependent margins to push minority class decision
    boundaries further from the origin.
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        max_margin: float = 0.5,
        scale: float = 30.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            samples_per_class: List of sample counts for each class
            max_margin: Maximum margin value
            scale: Scaling factor for logits
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.scale = scale
        self.reduction = reduction
        
        # Compute class-dependent margins
        # Margin is inversely proportional to n^(1/4)
        margins = max_margin / np.power(samples_per_class, 0.25)
        margins = margins / margins.max() * max_margin
        
        self.register_buffer('margins', torch.tensor(margins, dtype=torch.float32))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B, C) multi-hot labels
        """
        # Apply margins to positive class logits (move to same device as inputs)
        margins = self.margins.to(inputs.device).unsqueeze(0)
        
        # For positive samples, subtract margin (makes it harder to classify)
        # This forces the model to learn more discriminative features
        adjusted_inputs = inputs - margins * targets
        
        # Scale and compute BCE
        loss = F.binary_cross_entropy_with_logits(
            self.scale * adjusted_inputs, 
            targets, 
            reduction=self.reduction
        )
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that includes:
    1. Main task loss (Asymmetric or Class-Balanced Focal)
    2. Auxiliary losses (for deep supervision)
    3. Optional label smoothing
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        loss_type: str = 'asymmetric',  # 'asymmetric', 'class_balanced', 'focal', 'ldam'
        use_label_smoothing: bool = True,
        aux_weight: float = 0.3,
        gamma: float = 2.0,
        **kwargs
    ):
        """
        Args:
            samples_per_class: List of sample counts for each class
            loss_type: Type of main loss function
            use_label_smoothing: Whether to apply label smoothing
            aux_weight: Weight for auxiliary losses
            gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.aux_weight = aux_weight
        self.use_label_smoothing = use_label_smoothing
        
        # Label smoothing
        if use_label_smoothing:
            self.label_smoother = AsymmetricLabelSmoothing(
                samples_per_class=samples_per_class,
                smoothing_pos=0.1,
                smoothing_neg=0.02,
                adaptive=True
            )
        
        # Main loss
        if loss_type == 'asymmetric':
            self.main_loss = AsymmetricLoss(
                gamma_neg=4.0,
                gamma_pos=1.0,
                clip=0.05
            )
        elif loss_type == 'class_balanced':
            self.main_loss = ClassBalancedFocalLoss(
                samples_per_class=samples_per_class,
                beta=0.9999,
                gamma=gamma
            )
        elif loss_type == 'ldam':
            self.main_loss = MultiLabelLDAMLoss(
                samples_per_class=samples_per_class,
                max_margin=0.5,
                scale=30.0
            )
        else:  # focal
            self.main_loss = FocalLoss(gamma=gamma)
        
        # Auxiliary loss (simpler, no class balancing)
        self.aux_loss = FocalLoss(gamma=gamma)
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict with 'main', 'aux_clinical', 'aux_dermoscopic' logits
            targets: (B, C) multi-hot labels
        
        Returns:
            Dict with 'total', 'main', 'aux_clinical', 'aux_dermoscopic' losses
        """
        # Apply label smoothing if enabled
        if self.use_label_smoothing:
            smoothed_targets = self.label_smoother.smooth_labels(targets)
        else:
            smoothed_targets = targets
        
        # Main loss
        main_loss = self.main_loss(outputs['main'], smoothed_targets)
        
        losses = {'main': main_loss}
        total_loss = main_loss
        
        # Auxiliary losses (if present)
        if 'aux_clinical' in outputs:
            aux_clinical_loss = self.aux_loss(outputs['aux_clinical'], targets)
            losses['aux_clinical'] = aux_clinical_loss
            total_loss = total_loss + self.aux_weight * aux_clinical_loss
        
        if 'aux_dermoscopic' in outputs:
            aux_dermoscopic_loss = self.aux_loss(outputs['aux_dermoscopic'], targets)
            losses['aux_dermoscopic'] = aux_dermoscopic_loss
            total_loss = total_loss + self.aux_weight * aux_dermoscopic_loss
        
        losses['total'] = total_loss
        
        return losses


def get_samples_per_class(class_weights: Dict[str, float], total_samples: int) -> List[int]:
    """
    Estimate samples per class from class weights.
    
    Args:
        class_weights: Dict mapping class name to weight (inverse frequency)
        total_samples: Total number of training samples
    
    Returns:
        List of estimated sample counts per class
    """
    # Weights are typically inverse frequency, so invert them
    weights = np.array(list(class_weights.values()))
    
    # Normalize to get frequencies
    inv_weights = 1.0 / weights
    frequencies = inv_weights / inv_weights.sum()
    
    # Estimate sample counts
    samples = (frequencies * total_samples).astype(int)
    
    return samples.tolist()


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 4
    num_classes = 11
    
    # Simulated samples per class (imbalanced)
    samples_per_class = [1000, 800, 500, 400, 300, 200, 150, 100, 80, 50, 30]
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print(f"\nInput shapes: logits={logits.shape}, targets={targets.shape}")
    
    # Test Focal Loss
    focal = FocalLoss(gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"\n1. Focal Loss: {focal_loss.item():.4f}")
    
    # Test Class-Balanced Focal Loss
    cb_focal = ClassBalancedFocalLoss(samples_per_class=samples_per_class, gamma=2.0)
    cb_loss = cb_focal(logits, targets)
    print(f"2. Class-Balanced Focal Loss: {cb_loss.item():.4f}")
    
    # Test Asymmetric Loss
    asym = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    asym_loss = asym(logits, targets)
    print(f"3. Asymmetric Loss: {asym_loss.item():.4f}")
    
    # Test LDAM Loss
    ldam = MultiLabelLDAMLoss(samples_per_class=samples_per_class)
    ldam_loss = ldam(logits, targets)
    print(f"4. Multi-Label LDAM Loss: {ldam_loss.item():.4f}")
    
    # Test Combined Loss
    outputs = {
        'main': logits,
        'aux_clinical': torch.randn(batch_size, num_classes),
        'aux_dermoscopic': torch.randn(batch_size, num_classes)
    }
    
    combined = CombinedLoss(
        samples_per_class=samples_per_class,
        loss_type='asymmetric',
        use_label_smoothing=True,
        aux_weight=0.3
    )
    combined_losses = combined(outputs, targets)
    
    print(f"\n5. Combined Loss:")
    for name, value in combined_losses.items():
        print(f"   {name}: {value.item():.4f}")
    
    print("\n✅ All loss functions tested successfully!")

