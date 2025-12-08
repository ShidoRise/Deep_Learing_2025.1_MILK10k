"""
EfficientNetV2-L Model Architecture for MILK10k Skin Lesion Classification
Enhanced with Cross-Modal Attention Fusion for improved Macro F1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import *


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape
        y = self.excitation(x)
        return x * y


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for fusing clinical and dermoscopic features.
    Each modality attends to the other to capture inter-modal relationships.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for clinical features
        self.clinical_q = nn.Linear(feature_dim, feature_dim)
        self.clinical_k = nn.Linear(feature_dim, feature_dim)
        self.clinical_v = nn.Linear(feature_dim, feature_dim)
        
        # Query, Key, Value projections for dermoscopic features
        self.dermoscopic_q = nn.Linear(feature_dim, feature_dim)
        self.dermoscopic_k = nn.Linear(feature_dim, feature_dim)
        self.dermoscopic_v = nn.Linear(feature_dim, feature_dim)
        
        # Output projections
        self.clinical_out = nn.Linear(feature_dim, feature_dim)
        self.dermoscopic_out = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.norm_clinical = nn.LayerNorm(feature_dim)
        self.norm_dermoscopic = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        clinical_features: torch.Tensor, 
        dermoscopic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clinical_features: (B, D) clinical image features
            dermoscopic_features: (B, D) dermoscopic image features
        Returns:
            Tuple of enhanced (clinical, dermoscopic) features
        """
        B = clinical_features.shape[0]
        
        # Add sequence dimension for attention: (B, D) -> (B, 1, D)
        clinical = clinical_features.unsqueeze(1)
        dermoscopic = dermoscopic_features.unsqueeze(1)
        
        # Clinical attends to dermoscopic (cross-attention)
        q_c = self.clinical_q(clinical)
        k_d = self.dermoscopic_k(dermoscopic)
        v_d = self.dermoscopic_v(dermoscopic)
        
        # Compute attention: clinical query attending to dermoscopic key/value
        attn_c = torch.matmul(q_c, k_d.transpose(-2, -1)) * self.scale
        attn_c = F.softmax(attn_c, dim=-1)
        attn_c = self.dropout(attn_c)
        clinical_enhanced = torch.matmul(attn_c, v_d)
        clinical_enhanced = self.clinical_out(clinical_enhanced)
        
        # Dermoscopic attends to clinical (cross-attention)
        q_d = self.dermoscopic_q(dermoscopic)
        k_c = self.clinical_k(clinical)
        v_c = self.clinical_v(clinical)
        
        attn_d = torch.matmul(q_d, k_c.transpose(-2, -1)) * self.scale
        attn_d = F.softmax(attn_d, dim=-1)
        attn_d = self.dropout(attn_d)
        dermoscopic_enhanced = torch.matmul(attn_d, v_c)
        dermoscopic_enhanced = self.dermoscopic_out(dermoscopic_enhanced)
        
        # Residual connections with layer norm
        clinical_out = self.norm_clinical(clinical + clinical_enhanced).squeeze(1)
        dermoscopic_out = self.norm_dermoscopic(dermoscopic + dermoscopic_enhanced).squeeze(1)
        
        return clinical_out, dermoscopic_out


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to combine cross-attended features.
    Learns to weight the contribution of each modality.
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Projection for combined features
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
    
    def forward(
        self, 
        clinical_features: torch.Tensor, 
        dermoscopic_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            clinical_features: (B, D)
            dermoscopic_features: (B, D)
        Returns:
            Fused features (B, D)
        """
        combined = torch.cat([clinical_features, dermoscopic_features], dim=1)
        
        # Compute gating weights
        gates = self.gate(combined)  # (B, 2)
        
        # Weighted combination
        weighted = (
            gates[:, 0:1] * clinical_features + 
            gates[:, 1:2] * dermoscopic_features
        )
        
        # Project concatenated features and add weighted sum
        projected = self.projection(combined)
        
        return projected + weighted


class MetadataEncoder(nn.Module):
    """
    Enhanced metadata encoder with SE blocks for better feature integration.
    """
    
    def __init__(
        self, 
        input_dim: int = 18, 
        hidden_dim: int = 128, 
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        self.se = SqueezeExcitation(hidden_dim, reduction=8)
        
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )
    
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        x = self.encoder(metadata)
        x = self.se(x)
        return self.output(x)


class SkinLesionClassifierV2(nn.Module):
    """
    EfficientNetV2-L based classifier with Cross-Modal Attention Fusion.
    
    Architecture:
    1. Dual EfficientNetV2-L backbones for clinical and dermoscopic images
    2. Cross-modal attention between image modalities
    3. Gated fusion for combining features
    4. Enhanced metadata integration with SE blocks
    5. Optional auxiliary heads for deep supervision
    
    Optimized for A100 80GB GPU with 480x480 images.
    """
    
    def __init__(
        self,
        architecture: str = 'tf_efficientnetv2_l',
        num_classes: int = 11,
        pretrained: bool = True,
        use_metadata: bool = True,
        metadata_dim: int = 18,
        dropout: float = 0.3,
        use_auxiliary_heads: bool = True,
        cross_attention_heads: int = 8
    ):
        super().__init__()
        
        self.use_metadata = use_metadata
        self.use_auxiliary_heads = use_auxiliary_heads
        self.num_classes = num_classes
        
        # Dual backbones - EfficientNetV2-L
        self.clinical_backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        self.dermoscopic_backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # Get feature dimension from backbone
        feature_dim = self.clinical_backbone.num_features
        self.feature_dim = feature_dim
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            feature_dim=feature_dim,
            num_heads=cross_attention_heads,
            dropout=dropout
        )
        
        # Gated fusion
        self.gated_fusion = GatedFusion(feature_dim)
        
        # Metadata encoder
        if use_metadata:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                hidden_dim=128,
                output_dim=64,
                dropout=dropout
            )
            classifier_input_dim = feature_dim + 64
        else:
            classifier_input_dim = feature_dim
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary heads for deep supervision (optional)
        if use_auxiliary_heads:
            self.aux_clinical = nn.Linear(feature_dim, num_classes)
            self.aux_dermoscopic = nn.Linear(feature_dim, num_classes)
    
    def forward(
        self, 
        images: Tuple[torch.Tensor, torch.Tensor], 
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            images: Tuple of (clinical_img, dermoscopic_img), each (B, 3, H, W)
            metadata: Optional metadata tensor (B, metadata_dim)
        
        Returns:
            logits: (B, num_classes)
            If training with auxiliary heads, returns dict with 'main', 'aux_clinical', 'aux_dermoscopic'
        """
        clinical_img, dermoscopic_img = images
        
        # Extract backbone features
        clinical_features = self.clinical_backbone(clinical_img)
        dermoscopic_features = self.dermoscopic_backbone(dermoscopic_img)
        
        # Cross-modal attention
        clinical_enhanced, dermoscopic_enhanced = self.cross_attention(
            clinical_features, dermoscopic_features
        )
        
        # Gated fusion
        fused_features = self.gated_fusion(clinical_enhanced, dermoscopic_enhanced)
        
        # Add metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            fused_features = torch.cat([fused_features, metadata_features], dim=1)
        
        # Main classification
        logits = self.classifier(fused_features)
        
        # Return auxiliary outputs during training
        if self.training and self.use_auxiliary_heads:
            aux_clinical = self.aux_clinical(clinical_features)
            aux_dermoscopic = self.aux_dermoscopic(dermoscopic_features)
            return {
                'main': logits,
                'aux_clinical': aux_clinical,
                'aux_dermoscopic': aux_dermoscopic
            }
        
        return logits


class SkinLesionClassifierV2Medium(SkinLesionClassifierV2):
    """
    EfficientNetV2-M variant for faster training or memory constraints.
    ~54M parameters per backbone vs ~120M for V2-L
    """
    
    def __init__(self, **kwargs):
        kwargs['architecture'] = 'tf_efficientnetv2_m'
        super().__init__(**kwargs)


class SkinLesionClassifierV2Small(SkinLesionClassifierV2):
    """
    EfficientNetV2-S variant for quick experiments.
    ~21M parameters per backbone
    """
    
    def __init__(self, **kwargs):
        kwargs['architecture'] = 'tf_efficientnetv2_s'
        super().__init__(**kwargs)


def create_model_v2(
    architecture: str = 'tf_efficientnetv2_l',
    num_classes: int = 11,
    pretrained: bool = True,
    use_metadata: bool = True,
    metadata_dim: int = 18,
    dropout: float = 0.3,
    use_auxiliary_heads: bool = True
) -> nn.Module:
    """
    Factory function to create EfficientNetV2 models.
    
    Args:
        architecture: One of 'tf_efficientnetv2_s', 'tf_efficientnetv2_m', 'tf_efficientnetv2_l'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        use_metadata: Whether to use metadata features
        metadata_dim: Dimension of metadata input
        dropout: Dropout rate
        use_auxiliary_heads: Whether to use auxiliary classification heads
    
    Returns:
        Model instance
    """
    model = SkinLesionClassifierV2(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
        dropout=dropout,
        use_auxiliary_heads=use_auxiliary_heads
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing EfficientNetV2-L model creation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model_v2(
        architecture='tf_efficientnetv2_l',
        num_classes=11,
        pretrained=True,
        use_metadata=True,
        metadata_dim=18,
        dropout=0.3,
        use_auxiliary_heads=True
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    clinical_img = torch.randn(batch_size, 3, 480, 480).to(device)
    dermoscopic_img = torch.randn(batch_size, 3, 480, 480).to(device)
    metadata = torch.randn(batch_size, 18).to(device)
    
    # Training mode (returns dict with auxiliary outputs)
    model.train()
    with torch.no_grad():
        output = model((clinical_img, dermoscopic_img), metadata)
    
    print(f"\nTraining mode output:")
    print(f"  Main logits shape: {output['main'].shape}")
    print(f"  Aux clinical shape: {output['aux_clinical'].shape}")
    print(f"  Aux dermoscopic shape: {output['aux_dermoscopic'].shape}")
    
    # Eval mode (returns single tensor)
    model.eval()
    with torch.no_grad():
        output = model((clinical_img, dermoscopic_img), metadata)
    
    print(f"\nEval mode output:")
    print(f"  Logits shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Memory estimation for A100
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = model((clinical_img, dermoscopic_img), metadata)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"  Peak GPU memory (batch=2): {peak_memory:.2f} GB")
    
    print("\n✅ Model test completed successfully!")

