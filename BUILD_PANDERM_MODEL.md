# Building the Tri-Modal PanDerm Fusion Model

A comprehensive step-by-step guide for implementing the Tri-Modal PanDerm Fusion Network for the MILK10k Skin Lesion Classification challenge.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Design](#2-architecture-design)
3. [Prerequisites](#3-prerequisites)
4. [Step 1: PanDerm/DermLIP Backbone Setup](#step-1-pandermdermLIP-backbone-setup)
5. [Step 2: Tri-Modal Cross-Attention (TMCT)](#step-2-tri-modal-cross-attention-tmct)
6. [Step 3: MONET Concept Embedding](#step-3-monet-concept-embedding)
7. [Step 4: Compound Loss Functions](#step-4-compound-loss-functions)
8. [Step 5: Training Pipeline](#step-5-training-pipeline)
9. [Step 6: XGBoost Hybrid Stacking](#step-6-xgboost-hybrid-stacking)
10. [Step 7: Ensemble and Inference](#step-7-ensemble-and-inference)
11. [Hardware Optimization](#hardware-optimization)
12. [Failure Modes and Mitigations](#failure-modes-and-mitigations)

---

## 1. Overview

### Problem Statement

The MILK10k challenge requires multi-modal diagnosis integrating:
- **Clinical close-up images**: Macroscopic view with 3D topography, surrounding skin context
- **Dermoscopic images**: Microscopic view with subsurface structures (pigment networks, vessels)
- **MONET semantic scores**: 11 probability scores for medical concepts (ulceration, vessels, erythema, etc.)
- **Patient metadata**: Age, sex, skin tone, anatomical site

### Primary Metric

**Macro F1 Score** - treats all 11 classes equally, penalizing models that ignore rare classes.

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Tri-Modal PanDerm Fusion Network                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐          │
│  │   Clinical   │    │  Dermoscopic │    │   MONET Scores (11)  │          │
│  │    Image     │    │    Image     │    │   + Metadata (7)     │          │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘          │
│         │                   │                       │                       │
│         ▼                   ▼                       ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐          │
│  │   DermLIP    │    │   DermLIP    │    │    MLP Projection    │          │
│  │   Encoder    │    │   Encoder    │    │   (11 → K tokens)    │          │
│  │   (ViT-L)    │    │   (ViT-L)    │    └──────────┬───────────┘          │
│  └──────┬───────┘    └──────┬───────┘               │                       │
│         │                   │                       │                       │
│         └─────────┬─────────┘                       │                       │
│                   ▼                                 │                       │
│         ┌─────────────────────┐                     │                       │
│         │  Stage 1: View      │                     │                       │
│         │  Alignment (Cross-  │                     │                       │
│         │  Attention)         │                     │                       │
│         └─────────┬───────────┘                     │                       │
│                   │                                 │                       │
│                   └────────────────┬────────────────┘                       │
│                                    ▼                                        │
│                   ┌─────────────────────────────────┐                       │
│                   │  Stage 2: Semantic Gating       │                       │
│                   │  (MONET queries visual tokens)  │                       │
│                   └─────────────────┬───────────────┘                       │
│                                     ▼                                       │
│                   ┌─────────────────────────────────┐                       │
│                   │  Stage 3: Global Context        │                       │
│                   │  Pooling (Learnable Query)      │                       │
│                   └─────────────────┬───────────────┘                       │
│                                     ▼                                       │
│                   ┌─────────────────────────────────┐                       │
│                   │  Hybrid Classification Head     │                       │
│                   │  (MLP + XGBoost Stacking)       │                       │
│                   └─────────────────┬───────────────┘                       │
│                                     ▼                                       │
│                            [11-class output]                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Design

### 2.1 Comparison with Current Implementation

| Component | Current (EfficientNet) | New (PanDerm) |
|-----------|------------------------|---------------|
| **Backbone** | EfficientNet-B3/B4 (CNN) | DermLIP ViT-L (Transformer) |
| **Parameters** | ~40M (dual) | ~300M (dual) |
| **Pre-training** | ImageNet (natural images) | 2M+ dermatology images |
| **Receptive Field** | Local (grows with depth) | Global (from layer 1) |
| **MONET Integration** | Late concatenation | Cross-attention semantic gating |
| **Loss Function** | Focal Loss | Soft F1 + Focal compound |

### 2.2 Why PanDerm/DermLIP?

1. **Domain-Specific Pre-training**: Learned on 2.1M skin disease images vs generic ImageNet
2. **Vision-Language Alignment**: Already optimized to understand dermatology concepts
3. **Global Attention**: ViT captures long-range dependencies (satellite lesions, skin context)
4. **MONET Compatibility**: Pre-trained with CLIP-style alignment, natural fit for MONET scores

### 2.3 Key Innovation: Tri-Modal Cross-Attention

Instead of simple concatenation, we use 3-stage deep fusion:

```
Stage 1 (View Alignment):
    Q = f_derm,  K = f_clin,  V = f_clin
    f_derm←clin = MultiHeadAttention(Q, K, V) + f_derm
    
Stage 2 (Semantic Gating):
    Q = f_derm←clin,  K = E_concept,  V = E_concept
    f_fused = MultiHeadAttention(Q, K, V) + f_derm←clin
    
Stage 3 (Global Pooling):
    f_global = LearnableQuery(f_fused)
```

---

## 3. Prerequisites

### 3.1 Hardware Requirements

| Hardware | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| **GPU** | RTX 3090 (24GB) | RTX 4090 (24GB) | A100 (80GB) |
| **Batch Size** | 8-16 | 24-32 | 96-128 |
| **Precision** | FP16 | FP16 | BF16 |
| **Training Time** | ~24 hours | ~12 hours | ~4 hours |

### 3.2 Software Dependencies

```bash
# Add to requirements.txt
transformers>=4.35.0
huggingface_hub>=0.19.0
open_clip_torch>=2.24.0
xgboost>=2.0.0
scipy>=1.11.0
```

### 3.3 Model Weights

Download PanDerm/DermLIP weights from HuggingFace:

```python
# Option 1: DermLIP with PanDerm encoder (Recommended)
MODEL_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"

# Option 2: DermLIP with ViT-B/16
MODEL_ID = "redlessone/DermLIP_ViT-B-16"

# Option 3: Standard PanDerm (if DermLIP unavailable)
MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
```

---

## Step 1: PanDerm/DermLIP Backbone Setup

### 1.1 File Structure

Create `src/models_panderm.py`:

```python
"""
Tri-Modal PanDerm Fusion Model for MILK10k Skin Lesion Classification
Based on: DermLIP + Tri-Modal Cross-Attention Transformer (TMCT)
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from huggingface_hub import hf_hub_download
import open_clip
```

### 1.2 DermLIP Encoder Wrapper

```python
class DermLIPEncoder(nn.Module):
    """
    DermLIP Vision Encoder wrapper.
    Outputs sequence of patch tokens + CLS token.
    """
    
    def __init__(
        self, 
        model_name: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_layers: int = 0,  # Number of early layers to freeze
        output_tokens: bool = True  # Return all tokens or just CLS
    ):
        super().__init__()
        
        # Load DermLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained=model_name
        )
        self.visual = self.model.visual
        
        # Freeze early layers for transfer learning
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        self.output_tokens = output_tokens
        self.embed_dim = self.visual.output_dim  # 768 for ViT-B
        
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer blocks"""
        # Freeze patch embedding
        for param in self.visual.conv1.parameters():
            param.requires_grad = False
        
        # Freeze transformer blocks
        for i, block in enumerate(self.visual.transformer.resblocks):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            If output_tokens=True: (B, L, D) all patch tokens
            If output_tokens=False: (B, D) CLS token only
        """
        # Get all tokens from ViT
        x = self.visual(x, output_tokens=self.output_tokens)
        return x
```

### 1.3 Dual-Stream PanDerm Backbone

```python
class DualStreamPanDerm(nn.Module):
    """
    Dual-stream PanDerm backbone for clinical and dermoscopic images.
    Separate encoders allow independent adaptation to each modality.
    """
    
    def __init__(
        self,
        model_name: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_clinical: int = 6,   # Freeze more for clinical (more noise)
        freeze_dermoscopic: int = 4  # Freeze less for dermoscopic (primary signal)
    ):
        super().__init__()
        
        # Two separate encoders (not shared weights)
        self.clinical_encoder = DermLIPEncoder(
            model_name=model_name,
            freeze_layers=freeze_clinical,
            output_tokens=True
        )
        self.dermoscopic_encoder = DermLIPEncoder(
            model_name=model_name,
            freeze_layers=freeze_dermoscopic,
            output_tokens=True
        )
        
        self.embed_dim = self.clinical_encoder.embed_dim
        
    def forward(
        self, 
        clinical_img: torch.Tensor, 
        dermoscopic_img: torch.Tensor
    ) -> tuple:
        """
        Args:
            clinical_img: (B, 3, 224, 224)
            dermoscopic_img: (B, 3, 224, 224)
        Returns:
            f_clin: (B, L, D) clinical tokens
            f_derm: (B, L, D) dermoscopic tokens
        """
        f_clin = self.clinical_encoder(clinical_img)
        f_derm = self.dermoscopic_encoder(dermoscopic_img)
        return f_clin, f_derm
```

### 1.4 Layer-Wise Learning Rate Decay

```python
def get_layer_wise_lr_params(model, base_lr=1e-4, decay_rate=0.9):
    """
    Create parameter groups with layer-wise learning rate decay.
    Earlier layers get lower learning rates to preserve pre-trained knowledge.
    
    Args:
        model: The model
        base_lr: Learning rate for the classification head
        decay_rate: Multiplicative decay per layer (0.9 = 10% reduction)
    
    Returns:
        List of parameter groups for optimizer
    """
    param_groups = []
    
    # Classification head - highest LR
    head_params = []
    backbone_params = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'classifier' in name or 'fusion' in name or 'concept' in name:
            head_params.append(param)
        else:
            # Extract layer number from name
            layer_num = None
            if 'resblocks' in name:
                # e.g., "visual.transformer.resblocks.5.mlp.c_fc.weight"
                parts = name.split('.')
                for i, p in enumerate(parts):
                    if p == 'resblocks':
                        layer_num = int(parts[i + 1])
                        break
            
            if layer_num is not None:
                if layer_num not in backbone_params:
                    backbone_params[layer_num] = []
                backbone_params[layer_num].append(param)
            else:
                # Embedding layers - lowest LR
                if -1 not in backbone_params:
                    backbone_params[-1] = []
                backbone_params[-1].append(param)
    
    # Add head params with base LR
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'name': 'head'
        })
    
    # Add backbone params with decaying LR
    max_layer = max(backbone_params.keys()) if backbone_params else 0
    for layer_num in sorted(backbone_params.keys(), reverse=True):
        depth = max_layer - layer_num + 1
        layer_lr = base_lr * (decay_rate ** depth)
        param_groups.append({
            'params': backbone_params[layer_num],
            'lr': layer_lr,
            'name': f'layer_{layer_num}'
        })
    
    return param_groups
```

---

## Step 2: Tri-Modal Cross-Attention (TMCT)

### 2.1 Multi-Head Cross-Attention Module

```python
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention where queries attend to key-value pairs.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,      # (B, L_q, D)
        key: torch.Tensor,        # (B, L_kv, D)
        value: torch.Tensor,      # (B, L_kv, D)
        attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, L_q, D = query.shape
        L_kv = key.shape[1]
        
        # Project
        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.out_proj(out)
        
        return out
```

### 2.2 TMCT Fusion Block

```python
class TMCTFusionBlock(nn.Module):
    """
    Tri-Modal Cross-Attention Transformer (TMCT) Block.
    Implements the 3-stage fusion process.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_concept_tokens: int = 11  # Number of MONET concepts
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Stage 1: View Alignment (dermoscopic queries clinical)
        self.view_align_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.view_align_norm1 = nn.LayerNorm(embed_dim)
        self.view_align_norm2 = nn.LayerNorm(embed_dim)
        
        # Stage 2: Semantic Gating (visual queries concepts)
        self.semantic_gate_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.semantic_gate_norm1 = nn.LayerNorm(embed_dim)
        self.semantic_gate_norm2 = nn.LayerNorm(embed_dim)
        
        # MLP for both stages
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        f_derm: torch.Tensor,    # (B, L, D) dermoscopic tokens
        f_clin: torch.Tensor,    # (B, L, D) clinical tokens
        f_concept: torch.Tensor  # (B, K, D) concept embeddings
    ) -> torch.Tensor:
        """
        3-stage fusion process.
        
        Returns:
            f_fused: (B, L, D) fused visual tokens
        """
        # Stage 1: View Alignment
        # Dermoscopic features attend to clinical features
        f_derm_aligned = self.view_align_norm1(f_derm)
        f_clin_norm = self.view_align_norm1(f_clin)
        
        cross_attn_out = self.view_align_attn(
            query=f_derm_aligned,
            key=f_clin_norm,
            value=f_clin_norm
        )
        f_derm_aligned = f_derm + cross_attn_out  # Residual
        f_derm_aligned = f_derm_aligned + self.mlp1(self.view_align_norm2(f_derm_aligned))
        
        # Stage 2: Semantic Gating
        # Visual features attend to concept embeddings
        f_visual = self.semantic_gate_norm1(f_derm_aligned)
        f_concept_norm = self.semantic_gate_norm1(f_concept)
        
        semantic_attn_out = self.semantic_gate_attn(
            query=f_visual,
            key=f_concept_norm,
            value=f_concept_norm
        )
        f_fused = f_derm_aligned + semantic_attn_out  # Residual
        f_fused = f_fused + self.mlp2(self.semantic_gate_norm2(f_fused))
        
        return f_fused
```

### 2.3 Global Context Pooling

```python
class GlobalContextPooling(nn.Module):
    """
    Learnable query-based pooling to summarize sequence into single vector.
    Similar to CLS token but with learned attention over all tokens.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        self.attention = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence of tokens
        Returns:
            (B, D) global descriptor
        """
        B = x.shape[0]
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)
        
        # Attend to sequence
        pooled = self.attention(query=query, key=x, value=x)
        pooled = self.norm(pooled)
        
        return pooled.squeeze(1)  # (B, D)
```

---

## Step 3: MONET Concept Embedding

### 3.1 Concept Projector

```python
class MONETConceptEmbedding(nn.Module):
    """
    Project MONET probability scores into concept token embeddings.
    Each of the 11 MONET scores becomes a separate token for attention.
    """
    
    def __init__(
        self,
        num_concepts: int = 11,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        
        # Learnable concept embeddings (like word embeddings)
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, embed_dim) * 0.02
        )
        
        # Score-to-weight projection
        self.score_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh()  # Scale modulation
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        monet_scores: torch.Tensor  # (B, 11) probability scores
    ) -> torch.Tensor:
        """
        Convert MONET scores to concept token embeddings.
        
        Returns:
            (B, 11, D) concept embeddings modulated by scores
        """
        B = monet_scores.shape[0]
        
        # Expand concept embeddings for batch
        concepts = self.concept_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, 11, D)
        
        # Project each score to modulation weights
        scores_expanded = monet_scores.unsqueeze(-1)  # (B, 11, 1)
        modulation = self.score_proj(scores_expanded)  # (B, 11, D)
        
        # Modulate concept embeddings by scores
        # High score -> strong concept signal, low score -> weak signal
        concepts = concepts * (1 + modulation)  # Multiplicative modulation
        
        concepts = self.norm(concepts)
        concepts = self.dropout(concepts)
        
        return concepts
```

### 3.2 MONET Concept Names (Reference)

```python
MONET_CONCEPTS = {
    0: 'ulceration_crust',      # Indicates malignancy (Melanoma, BCC)
    1: 'hair',                   # Artifact indicator
    2: 'vasculature_vessels',    # Vascular lesions, neo-angiogenesis
    3: 'erythema',               # Inflammation marker
    4: 'pigmented',              # Melanocytic vs non-melanocytic
    5: 'gel_fluid',              # Dermoscopy artifact
    6: 'skin_markings_pen',      # Artifact - should ignore
    7: 'blue_white_veil',        # Invasive melanoma marker
    8: 'pigment_network',        # Melanocytic lesion marker
    9: 'regression_structures',  # Melanoma regression
    10: 'atypical_network'       # Malignancy indicator
}
```

---

## Step 4: Compound Loss Functions

### 4.1 Differentiable Soft F1 Loss

```python
class SoftF1Loss(nn.Module):
    """
    Differentiable approximation of F1 score for direct metric optimization.
    Computed at batch level for stable gradients.
    
    Formula:
        Soft_TP = sum(p * y)
        Soft_FP = sum(p * (1-y))
        Soft_FN = sum((1-p) * y)
        Soft_F1 = 2*TP / (2*TP + FP + FN + eps)
    """
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(
        self, 
        logits: torch.Tensor,  # (B, C) raw logits
        targets: torch.Tensor  # (B, C) binary targets
    ) -> torch.Tensor:
        """
        Compute Macro Soft F1 Loss (1 - F1).
        """
        probs = torch.sigmoid(logits)
        
        # Per-class soft metrics
        soft_tp = (probs * targets).sum(dim=0)
        soft_fp = (probs * (1 - targets)).sum(dim=0)
        soft_fn = ((1 - probs) * targets).sum(dim=0)
        
        # Per-class F1
        soft_f1 = (2 * soft_tp) / (2 * soft_tp + soft_fp + soft_fn + self.epsilon)
        
        # Macro F1 (average across classes)
        macro_f1 = soft_f1.mean()
        
        # Return loss (1 - F1)
        return 1 - macro_f1
```

### 4.2 Weighted Focal Loss

```python
class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weights for handling class imbalance.
    
    Formula:
        FL = -alpha * (1-p)^gamma * log(p)  for positive class
        FL = -(1-alpha) * p^gamma * log(1-p)  for negative class
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(
        self,
        logits: torch.Tensor,  # (B, C)
        targets: torch.Tensor  # (B, C)
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Apply focal weight
        loss = focal_weight * bce
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = loss * weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

### 4.3 Compound Loss

```python
class CompoundLoss(nn.Module):
    """
    Combined Focal + Soft F1 Loss for balanced optimization.
    
    Total Loss = lambda1 * Focal + lambda2 * SoftF1
    """
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        focal_weight: float = 0.5,
        soft_f1_weight: float = 0.5
    ):
        super().__init__()
        
        self.focal = WeightedFocalLoss(gamma=focal_gamma, class_weights=class_weights)
        self.soft_f1 = SoftF1Loss()
        
        self.focal_weight = focal_weight
        self.soft_f1_weight = soft_f1_weight
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        focal_loss = self.focal(logits, targets)
        soft_f1_loss = self.soft_f1(logits, targets)
        
        total = self.focal_weight * focal_loss + self.soft_f1_weight * soft_f1_loss
        
        return {
            'total': total,
            'focal': focal_loss,
            'soft_f1': soft_f1_loss
        }
```

### 4.4 Auxiliary Loss for Deep Supervision

```python
class AuxiliaryLoss(nn.Module):
    """
    Compute main loss + auxiliary losses from intermediate features.
    
    Final = L_main + aux_weight * (L_clinical + L_dermoscopic)
    """
    
    def __init__(
        self,
        main_criterion: nn.Module,
        aux_weight: float = 0.3
    ):
        super().__init__()
        self.main_criterion = main_criterion
        self.aux_weight = aux_weight
        
    def forward(
        self,
        outputs: dict,  # {'main': logits, 'aux_clinical': logits, 'aux_dermoscopic': logits}
        targets: torch.Tensor
    ) -> dict:
        main_loss = self.main_criterion(outputs['main'], targets)
        
        losses = {'main': main_loss['total'] if isinstance(main_loss, dict) else main_loss}
        
        if 'aux_clinical' in outputs:
            aux_clin = self.main_criterion(outputs['aux_clinical'], targets)
            aux_clin = aux_clin['total'] if isinstance(aux_clin, dict) else aux_clin
            losses['aux_clinical'] = aux_clin
            
        if 'aux_dermoscopic' in outputs:
            aux_derm = self.main_criterion(outputs['aux_dermoscopic'], targets)
            aux_derm = aux_derm['total'] if isinstance(aux_derm, dict) else aux_derm
            losses['aux_dermoscopic'] = aux_derm
        
        # Total with auxiliary
        total = losses['main']
        if 'aux_clinical' in losses:
            total = total + self.aux_weight * losses['aux_clinical']
        if 'aux_dermoscopic' in losses:
            total = total + self.aux_weight * losses['aux_dermoscopic']
        
        losses['total'] = total
        return losses
```

---

## Step 5: Training Pipeline

### 5.1 Training Configuration

```python
# config.py additions
MODEL_CONFIG_PANDERM = {
    'model_name': 'redlessone/DermLIP_PanDerm-base-w-PubMed-256',
    'embed_dim': 768,
    'num_heads': 8,
    'num_classes': 11,
    'num_concept_tokens': 11,  # MONET concepts
    'dropout': 0.1,
    'freeze_clinical': 6,      # Freeze first 6 layers of clinical encoder
    'freeze_dermoscopic': 4,   # Freeze first 4 layers of dermoscopic encoder
}

TRAIN_CONFIG_PANDERM = {
    'batch_size': 32,          # A100: 96, RTX 4090: 32, RTX 3090: 16
    'num_epochs': 60,
    'base_lr': 1e-4,           # For head/fusion
    'backbone_lr_decay': 0.9,  # Layer-wise decay rate
    'min_lr': 1e-7,
    'weight_decay': 0.05,      # Higher for large ViT
    'warmup_epochs': 3,
    'gradient_clip': 1.0,
    'gradient_accumulation': 2,  # Effective batch = 64
    'mixed_precision': 'bf16',   # 'fp16' for consumer GPUs
    
    # Dropout strategies
    'modality_dropout': 0.2,   # Probability of zeroing clinical image
    'concept_dropout': 0.1,    # Probability of zeroing individual MONET scores
    
    # Loss weights
    'focal_weight': 0.5,
    'soft_f1_weight': 0.5,
    'aux_loss_weight': 0.3,
}

IMAGE_CONFIG_PANDERM = {
    'image_size': 224,  # ViT native resolution
    'normalize_mean': [0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
    'normalize_std': [0.26862954, 0.26130258, 0.27577711],
}
```

### 5.2 Modality & Concept Dropout

```python
def apply_modality_dropout(
    clinical_img: torch.Tensor,
    monet_scores: torch.Tensor,
    modality_drop_prob: float = 0.2,
    concept_drop_prob: float = 0.1,
    training: bool = True
) -> tuple:
    """
    Apply dropout to modalities during training for robustness.
    
    - Modality dropout: Zero out entire clinical image (force reliance on dermoscopy)
    - Concept dropout: Zero out random MONET scores (prevent over-reliance)
    """
    if not training:
        return clinical_img, monet_scores
    
    B = clinical_img.shape[0]
    device = clinical_img.device
    
    # Modality dropout - zero out clinical images
    if modality_drop_prob > 0:
        modality_mask = torch.rand(B, 1, 1, 1, device=device) > modality_drop_prob
        clinical_img = clinical_img * modality_mask.float()
    
    # Concept dropout - zero out individual MONET scores
    if concept_drop_prob > 0:
        concept_mask = torch.rand_like(monet_scores) > concept_drop_prob
        monet_scores = monet_scores * concept_mask.float()
    
    return clinical_img, monet_scores
```

### 5.3 Training Loop Skeleton

```python
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,  # GradScaler for mixed precision
    config,
    epoch
):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        clinical_img = batch['clinical_image'].cuda()
        dermoscopic_img = batch['dermoscopic_image'].cuda()
        monet_scores = batch['monet_scores'].cuda()
        metadata = batch['metadata'].cuda()
        targets = batch['targets'].cuda()
        
        # Apply dropout augmentation
        clinical_img, monet_scores = apply_modality_dropout(
            clinical_img, monet_scores,
            modality_drop_prob=config['modality_dropout'],
            concept_drop_prob=config['concept_dropout'],
            training=True
        )
        
        # Mixed precision forward
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(clinical_img, dermoscopic_img, monet_scores, metadata)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total'] / config['gradient_accumulation']
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config['gradient_accumulation']
    
    scheduler.step()
    return total_loss / len(train_loader)
```

---

## Step 6: XGBoost Hybrid Stacking

### 6.1 Feature Extraction

```python
def extract_features_for_xgboost(
    model,
    dataloader,
    device='cuda'
) -> tuple:
    """
    Extract frozen features from trained PanDerm model for XGBoost.
    
    Returns:
        features: numpy array (N, D + metadata_dim + monet_dim)
        labels: numpy array (N, num_classes)
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            clinical_img = batch['clinical_image'].to(device)
            dermoscopic_img = batch['dermoscopic_image'].to(device)
            monet_scores = batch['monet_scores'].to(device)
            metadata = batch['metadata'].to(device)
            targets = batch['targets']
            
            # Get intermediate features (before classifier)
            features = model.get_features(
                clinical_img, dermoscopic_img, monet_scores, metadata
            )
            logits = model(clinical_img, dermoscopic_img, monet_scores, metadata)
            
            # Concatenate: DL features + raw metadata + MONET + DL logits
            combined = torch.cat([
                features,
                metadata,
                monet_scores,
                torch.sigmoid(logits)  # DL predictions as features
            ], dim=1)
            
            all_features.append(combined.cpu().numpy())
            all_labels.append(targets.numpy())
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
    
    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_logits, axis=0)
    )
```

### 6.2 XGBoost Training

```python
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

def train_xgboost_stacking(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 11
) -> MultiOutputClassifier:
    """
    Train XGBoost classifier for each class (multi-label).
    """
    # XGBoost params optimized for stacking
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'tree_method': 'hist',  # Fast histogram-based
        'device': 'cuda'  # GPU acceleration
    }
    
    # Multi-output wrapper for multi-label
    base_model = xgb.XGBClassifier(**xgb_params)
    multi_model = MultiOutputClassifier(base_model, n_jobs=-1)
    
    # Fit
    multi_model.fit(train_features, train_labels)
    
    return multi_model
```

### 6.3 Ensemble Weight Optimization

```python
from scipy.optimize import minimize

def optimize_ensemble_weights(
    panderm_probs: np.ndarray,
    efficientnet_probs: np.ndarray,
    xgboost_probs: np.ndarray,
    true_labels: np.ndarray
) -> np.ndarray:
    """
    Find optimal ensemble weights using Nelder-Mead optimization.
    Maximizes Macro F1 on validation set.
    """
    def objective(weights):
        w1, w2, w3 = weights
        # Normalize weights
        total = w1 + w2 + w3
        w1, w2, w3 = w1/total, w2/total, w3/total
        
        # Weighted average
        ensemble_probs = w1 * panderm_probs + w2 * efficientnet_probs + w3 * xgboost_probs
        predictions = (ensemble_probs > 0.5).astype(int)
        
        # Compute Macro F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        
        return -macro_f1  # Negative because we minimize
    
    # Initial weights
    x0 = [0.4, 0.3, 0.3]
    
    # Bounds: each weight between 0 and 1
    bounds = [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
    
    # Optimize
    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    
    # Normalize final weights
    weights = result.x
    weights = weights / weights.sum()
    
    print(f"Optimal weights: PanDerm={weights[0]:.3f}, EfficientNet={weights[1]:.3f}, XGBoost={weights[2]:.3f}")
    print(f"Best Macro F1: {-result.fun:.4f}")
    
    return weights
```

---

## Step 7: Ensemble and Inference

### 7.1 Ensemble Model Class

```python
class EnsembleModel:
    """
    Ensemble combining PanDerm, EfficientNet, and XGBoost.
    """
    
    def __init__(
        self,
        panderm_model,
        efficientnet_model,
        xgboost_model,
        weights: np.ndarray = None,
        device: str = 'cuda'
    ):
        self.panderm = panderm_model.to(device).eval()
        self.efficientnet = efficientnet_model.to(device).eval()
        self.xgboost = xgboost_model
        self.device = device
        
        # Default equal weights if not provided
        self.weights = weights if weights is not None else np.array([0.4, 0.3, 0.3])
        
    @torch.no_grad()
    def predict(
        self,
        clinical_img: torch.Tensor,
        dermoscopic_img: torch.Tensor,
        monet_scores: torch.Tensor,
        metadata: torch.Tensor
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        """
        # PanDerm prediction
        panderm_logits = self.panderm(
            clinical_img.to(self.device),
            dermoscopic_img.to(self.device),
            monet_scores.to(self.device),
            metadata.to(self.device)
        )
        panderm_probs = torch.sigmoid(panderm_logits).cpu().numpy()
        
        # EfficientNet prediction
        efficientnet_logits = self.efficientnet(
            (clinical_img.to(self.device), dermoscopic_img.to(self.device)),
            metadata.to(self.device)
        )
        efficientnet_probs = torch.sigmoid(efficientnet_logits).cpu().numpy()
        
        # Extract features for XGBoost
        panderm_features = self.panderm.get_features(
            clinical_img.to(self.device),
            dermoscopic_img.to(self.device),
            monet_scores.to(self.device),
            metadata.to(self.device)
        ).cpu().numpy()
        
        xgb_input = np.concatenate([
            panderm_features,
            metadata.cpu().numpy(),
            monet_scores.cpu().numpy(),
            panderm_probs
        ], axis=1)
        
        xgboost_probs = self.xgboost.predict_proba(xgb_input)
        # Convert list of arrays to single array
        xgboost_probs = np.stack([p[:, 1] for p in xgboost_probs], axis=1)
        
        # Weighted ensemble
        ensemble_probs = (
            self.weights[0] * panderm_probs +
            self.weights[1] * efficientnet_probs +
            self.weights[2] * xgboost_probs
        )
        
        return ensemble_probs
```

### 7.2 Test-Time Augmentation (TTA)

```python
def predict_with_tta(
    ensemble: EnsembleModel,
    clinical_img: torch.Tensor,
    dermoscopic_img: torch.Tensor,
    monet_scores: torch.Tensor,
    metadata: torch.Tensor,
    num_augments: int = 8
) -> np.ndarray:
    """
    Apply TTA: rotations (0, 90, 180, 270) x flips (none, horizontal).
    """
    all_preds = []
    
    augments = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, 2, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, 3, [2, 3]), [3]),
    ]
    
    for aug in augments[:num_augments]:
        aug_clinical = aug(clinical_img)
        aug_dermoscopic = aug(dermoscopic_img)
        
        pred = ensemble.predict(aug_clinical, aug_dermoscopic, monet_scores, metadata)
        all_preds.append(pred)
    
    # Average predictions
    return np.mean(all_preds, axis=0)
```

---

## Hardware Optimization

### A100 80GB Configuration

```python
# Maximum utilization config for A100
A100_CONFIG = {
    'batch_size': 96,
    'gradient_accumulation': 1,  # No accumulation needed
    'mixed_precision': 'bf16',   # BF16 stable for ViT
    'num_workers': 8,
    'pin_memory': True,
    'compile_model': True,       # torch.compile for 2.0+
}
```

### Consumer GPU (RTX 3090/4090)

```python
# Memory-efficient config
CONSUMER_CONFIG = {
    'batch_size': 16,            # RTX 3090
    'gradient_accumulation': 6,  # Effective batch = 96
    'mixed_precision': 'fp16',   # FP16 with loss scaling
    'num_workers': 4,
    'pin_memory': True,
    'activation_checkpointing': True,  # Trade compute for memory
}
```

### Activation Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientTMCT(TMCTFusionBlock):
    """TMCT with activation checkpointing for memory efficiency."""
    
    def forward(self, f_derm, f_clin, f_concept):
        # Checkpoint each stage
        f_aligned = checkpoint(self._stage1, f_derm, f_clin)
        f_fused = checkpoint(self._stage2, f_aligned, f_concept)
        return f_fused
    
    def _stage1(self, f_derm, f_clin):
        # View alignment
        ...
    
    def _stage2(self, f_aligned, f_concept):
        # Semantic gating
        ...
```

---

## Failure Modes and Mitigations

### Failure Mode 1: Clinical Dominance Trap

**Problem**: Model learns shortcuts from clinical image backgrounds (rulers, skin marks).

**Mitigation**:
```python
# Modality dropout - force dermoscopy reliance
clinical_img = clinical_img * (torch.rand(B, 1, 1, 1) > 0.2).float()
```

### Failure Mode 2: Rare Class Collapse

**Problem**: Model never predicts Dermatofibroma or VASC due to low frequency.

**Mitigation**:
```python
# Oversampling in DataLoader
from torch.utils.data import WeightedRandomSampler

sample_weights = compute_sample_weights(labels)  # Inverse frequency
sampler = WeightedRandomSampler(sample_weights, len(dataset))

# Post-hoc logit adjustment
logit_adjustment = torch.log(class_priors)
adjusted_logits = logits - logit_adjustment
```

### Failure Mode 3: MONET Overfitting

**Problem**: Model relies solely on MONET scores, ignoring visual evidence.

**Mitigation**:
```python
# Concept dropout
monet_scores = monet_scores * (torch.rand_like(monet_scores) > 0.1).float()
```

### Failure Mode 4: ViT Training Instability

**Problem**: Large ViT with small dataset causes gradient explosions.

**Mitigation**:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Layer-wise LR decay
# Early layers: 1e-6, Late layers: 1e-4

# Warmup
# Linear warmup for 3 epochs before cosine decay
```

---

## Training Checklist

- [ ] Download DermLIP weights from HuggingFace
- [ ] Implement `DualStreamPanDerm` backbone
- [ ] Implement `TMCTFusionBlock` with 3-stage attention
- [ ] Implement `MONETConceptEmbedding`
- [ ] Implement `CompoundLoss` (Focal + Soft F1)
- [ ] Implement `AuxiliaryLoss` for deep supervision
- [ ] Create training script with layer-wise LR decay
- [ ] Add modality/concept dropout augmentation
- [ ] Train PanDerm model (~60 epochs)
- [ ] Extract features for XGBoost
- [ ] Train XGBoost stacking model
- [ ] Optimize ensemble weights on validation set
- [ ] Generate final predictions with TTA
- [ ] Create submission file

---

## Expected Performance

| Model | Macro F1 (Validation) | Notes |
|-------|----------------------|-------|
| EfficientNet-B3 (baseline) | 0.68-0.72 | Current implementation |
| PanDerm (single) | 0.74-0.78 | Foundation model advantage |
| PanDerm + XGBoost | 0.76-0.80 | Tabular data boost |
| Full Ensemble | 0.78-0.82 | ViT + CNN diversity |

---

## References

1. [MILK10k Challenge](https://challenge.isic-archive.com/landing/milk10k/)
2. [PanDerm Paper](https://arxiv.org/html/2410.15038v2)
3. [DermLIP on HuggingFace](https://huggingface.co/redlessone/DermLIP_ViT-B-16)
4. [SkinM2Former (TMCT)](https://openaccess.thecvf.com/content/WACV2025/papers/Zhang_A_Novel_Perspective_for_Multi-Modal_Multi-Label_Skin_Lesion_Classification_WACV_2025_paper.pdf)
5. [Soft F1 Loss](https://arxiv.org/pdf/2108.10566)

