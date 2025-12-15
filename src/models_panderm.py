"""
Tri-Modal PanDerm Fusion Model for MILK10k Skin Lesion Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import sys
from pathlib import Path
import warnings

sys.path.append(str(Path(__file__).parent))
from config import MODEL_CONFIG_PANDERM, IMAGE_CONFIG_PANDERM

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class DermLIPEncoder(nn.Module):
    """DermLIP Vision Encoder wrapper with layer freezing support."""
    
    def __init__(
        self, 
        model_name: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_layers: int = 0,
        output_tokens: bool = True,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_tokens = output_tokens
        self.freeze_layers = freeze_layers
        
        if OPEN_CLIP_AVAILABLE:
            self._init_open_clip(model_name, pretrained)
        elif TIMM_AVAILABLE:
            self._init_timm_fallback(pretrained)
        else:
            raise ImportError("Either open_clip or timm must be installed")
        
        # Apply layer freezing
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _init_open_clip(self, model_name: str, pretrained: bool):
        loaded_successfully = False
        
        if pretrained and '/' in model_name:
            # HuggingFace model - use manual weight loading
            # (Native loading fails due to 'pretrain_path' in HF config)
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                
                print(f"Loading DermLIP from HuggingFace: {model_name}")
                
                # Download weights file
                weights_path = hf_hub_download(
                    repo_id=model_name,
                    filename="open_clip_model.safetensors"
                )
                
                # Build base ViT-B/16 (same architecture as DermLIP visual encoder)
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16',
                    pretrained=None
                )
                
                # Load weights from safetensors
                state_dict = load_file(weights_path)
                
                # Extract and load visual encoder weights
                visual_state = {
                    k.replace("visual.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("visual.")
                }
                
                if visual_state:
                    missing, unexpected = self.model.visual.load_state_dict(
                        visual_state, strict=False
                    )
                    print(f"✅ Loaded DermLIP visual encoder from HuggingFace")
                    print(f"   Weights loaded: {len(visual_state)} tensors")
                    if missing:
                        print(f"   Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"   Unexpected keys: {len(unexpected)}")
                else:
                    raise RuntimeError("No visual.* keys found in downloaded weights")
                
                self.visual = self.model.visual
                self.embed_dim = self.visual.output_dim
                self.num_patches = 196
                self.using_open_clip = True
                loaded_successfully = True
                
            except Exception as e:
                warnings.warn(
                    f"Failed to load DermLIP from HuggingFace: {e}. "
                    f"Falling back to timm ViT."
                )
        else:
            # Standard open_clip model or no pretrained weights
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16',
                    pretrained=model_name if pretrained else None
                )
                self.visual = self.model.visual
                self.embed_dim = self.visual.output_dim
                self.num_patches = 196
                self.using_open_clip = True
                loaded_successfully = True
                if pretrained:
                    print(f"✅ Loaded ViT-B-16 with pretrained={model_name}")
                else:
                    print(f"✅ Created ViT-B-16 without pretrained weights")
            except Exception as e:
                warnings.warn(f"Failed to load open_clip model: {e}. Falling back to timm ViT.")
        
        if not loaded_successfully:
            self._init_timm_fallback(pretrained)
    
    def _init_timm_fallback(self, pretrained: bool):
        self.visual = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        self.embed_dim = self.visual.embed_dim
        self.num_patches = 196
        self.using_open_clip = False
        self.model = None
        self.preprocess = None
    
    def _freeze_layers(self, num_layers: int):
        if self.using_open_clip:
            # Freeze patch embedding
            for param in self.visual.conv1.parameters():
                param.requires_grad = False
            if hasattr(self.visual, 'positional_embedding'):
                self.visual.positional_embedding.requires_grad = False
            
            # Freeze transformer blocks
            if hasattr(self.visual, 'transformer') and hasattr(self.visual.transformer, 'resblocks'):
                for i, block in enumerate(self.visual.transformer.resblocks):
                    if i < num_layers:
                        for param in block.parameters():
                            param.requires_grad = False
        else:
            # timm ViT structure
            for param in self.visual.patch_embed.parameters():
                param.requires_grad = False
            if hasattr(self.visual, 'pos_embed'):
                self.visual.pos_embed.requires_grad = False
            
            for i, block in enumerate(self.visual.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_open_clip:
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            cls_token = self.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            cls_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.visual.positional_embedding.unsqueeze(0)
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)
            
            if self.output_tokens:
                return x
            else:
                return self.visual.ln_post(x[:, 0, :])
        else:
            x = self.visual.forward_features(x)
            if self.output_tokens:
                return x
            else:
                return x[:, 0, :]
    
    def get_num_layers(self) -> int:
        if self.using_open_clip:
            return len(self.visual.transformer.resblocks)
        else:
            return len(self.visual.blocks)


class DualStreamPanDerm(nn.Module):
    """Dual-stream PanDerm backbone for clinical and dermoscopic images."""
    
    def __init__(
        self,
        model_name: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_clinical: int = 6,
        freeze_dermoscopic: int = 4,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.clinical_encoder = DermLIPEncoder(
            model_name=model_name,
            freeze_layers=freeze_clinical,
            output_tokens=True,
            pretrained=pretrained
        )
        
        self.dermoscopic_encoder = DermLIPEncoder(
            model_name=model_name,
            freeze_layers=freeze_dermoscopic,
            output_tokens=True,
            pretrained=pretrained
        )
        
        self.embed_dim = self.clinical_encoder.embed_dim
        self.num_patches = self.clinical_encoder.num_patches
    
    def forward(
        self, 
        clinical_img: torch.Tensor, 
        dermoscopic_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f_clin = self.clinical_encoder(clinical_img)
        f_derm = self.dermoscopic_encoder(dermoscopic_img)
        return f_clin, f_derm
    
    def get_clinical_cls(self, clinical_img: torch.Tensor) -> torch.Tensor:
        tokens = self.clinical_encoder(clinical_img)
        return tokens[:, 0, :]
    
    def get_dermoscopic_cls(self, dermoscopic_img: torch.Tensor) -> torch.Tensor:
        tokens = self.dermoscopic_encoder(dermoscopic_img)
        return tokens[:, 0, :]


class MONETConceptEmbedding(nn.Module):
    """Project MONET probability scores into concept token embeddings."""
    
    def __init__(
        self,
        num_concepts: int = 7,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, embed_dim) * 0.02
        )
        
        self.score_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh()
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, monet_scores: torch.Tensor) -> torch.Tensor:
        B = monet_scores.shape[0]
        concepts = self.concept_embeddings.unsqueeze(0).expand(B, -1, -1)
        scores_expanded = monet_scores.unsqueeze(-1)
        modulation = self.score_proj(scores_expanded)
        concepts = concepts * (1 + modulation)
        concepts = self.norm(concepts)
        concepts = self.dropout(concepts)
        return concepts


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention where queries attend to key-value pairs."""
    
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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L_q, D = query.shape
        L_kv = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.out_proj(out)
        return out


class TMCTFusionBlock(nn.Module):
    """
    Tri-Modal Cross-Attention Transformer Block.
    Stage 1: View Alignment (dermoscopic queries clinical)
    Stage 2: Semantic Gating (visual queries concepts)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.view_align_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.view_align_norm1 = nn.LayerNorm(embed_dim)
        self.view_align_norm2 = nn.LayerNorm(embed_dim)
        
        self.semantic_gate_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.semantic_gate_norm1 = nn.LayerNorm(embed_dim)
        self.semantic_gate_norm2 = nn.LayerNorm(embed_dim)
        
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
        f_derm: torch.Tensor,
        f_clin: torch.Tensor,
        f_concept: torch.Tensor
    ) -> torch.Tensor:
        # Stage 1: View Alignment
        f_derm_norm = self.view_align_norm1(f_derm)
        f_clin_norm = self.view_align_norm1(f_clin)
        cross_attn_out = self.view_align_attn(query=f_derm_norm, key=f_clin_norm, value=f_clin_norm)
        f_aligned = f_derm + cross_attn_out
        f_aligned = f_aligned + self.mlp1(self.view_align_norm2(f_aligned))
        
        # Stage 2: Semantic Gating
        f_visual_norm = self.semantic_gate_norm1(f_aligned)
        f_concept_norm = self.semantic_gate_norm1(f_concept)
        semantic_attn_out = self.semantic_gate_attn(query=f_visual_norm, key=f_concept_norm, value=f_concept_norm)
        f_fused = f_aligned + semantic_attn_out
        f_fused = f_fused + self.mlp2(self.semantic_gate_norm2(f_fused))
        return f_fused


class GlobalContextPooling(nn.Module):
    """Learnable query-based pooling to summarize sequence into global descriptor."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        pooled = self.attention(query=query, key=x, value=x)
        pooled = self.norm(pooled)
        return pooled.squeeze(1)


class MetadataEncoder(nn.Module):
    """Encode patient metadata into feature vector."""
    
    def __init__(
        self,
        input_dim: int = 11,  # 4 clinical + 7 MONET scores
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )
    
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.encoder(metadata)


class TriModalPanDermFusion(nn.Module):
    """Complete Tri-Modal PanDerm Fusion Network."""
    
    def __init__(
        self,
        model_name: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        embed_dim: int = 768,
        num_heads: int = 8,
        num_classes: int = 11,
        num_concept_tokens: int = 7,
        concept_hidden_dim: int = 256,
        tmct_num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        freeze_clinical: int = 6,
        freeze_dermoscopic: int = 4,
        use_auxiliary_heads: bool = True,
        pretrained: bool = True,
        use_metadata: bool = True,
        metadata_dim: int = 11  # Clinical features only, MONET goes through concept embedding
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_auxiliary_heads = use_auxiliary_heads
        self.use_metadata = use_metadata
        
        self.backbone = DualStreamPanDerm(
            model_name=model_name,
            freeze_clinical=freeze_clinical,
            freeze_dermoscopic=freeze_dermoscopic,
            pretrained=pretrained
        )
        
        self.concept_embedding = MONETConceptEmbedding(
            num_concepts=num_concept_tokens,
            embed_dim=embed_dim,
            hidden_dim=concept_hidden_dim,
            dropout=dropout
        )
        
        self.tmct_blocks = nn.ModuleList([
            TMCTFusionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(tmct_num_layers)
        ])
        
        self.global_pool = GlobalContextPooling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        if use_metadata:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                hidden_dim=128,
                output_dim=64,
                dropout=dropout
            )
            classifier_input_dim = embed_dim + 64
        else:
            classifier_input_dim = embed_dim
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        if use_auxiliary_heads:
            self.aux_clinical = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )
            self.aux_dermoscopic = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )
    
    def forward(
        self,
        clinical_img: torch.Tensor,
        dermoscopic_img: torch.Tensor,
        monet_scores: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        f_clin, f_derm = self.backbone(clinical_img, dermoscopic_img)
        clin_cls = f_clin[:, 0, :]
        derm_cls = f_derm[:, 0, :]
        f_concept = self.concept_embedding(monet_scores)
        
        f_fused = f_derm
        for tmct_block in self.tmct_blocks:
            f_fused = tmct_block(f_fused, f_clin, f_concept)
        
        f_global = self.global_pool(f_fused)
        
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            f_global = torch.cat([f_global, metadata_features], dim=1)
        
        logits = self.classifier(f_global)
        
        if self.training and self.use_auxiliary_heads:
            aux_clin_logits = self.aux_clinical(clin_cls)
            aux_derm_logits = self.aux_dermoscopic(derm_cls)
            return {
                'main': logits,
                'aux_clinical': aux_clin_logits,
                'aux_dermoscopic': aux_derm_logits
            }
        
        return logits
    
    def get_features(
        self,
        clinical_img: torch.Tensor,
        dermoscopic_img: torch.Tensor,
        monet_scores: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f_clin, f_derm = self.backbone(clinical_img, dermoscopic_img)
        f_concept = self.concept_embedding(monet_scores)
        
        f_fused = f_derm
        for tmct_block in self.tmct_blocks:
            f_fused = tmct_block(f_fused, f_clin, f_concept)
        
        f_global = self.global_pool(f_fused)
        
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            f_global = torch.cat([f_global, metadata_features], dim=1)
        
        return f_global


def get_layer_wise_lr_params(
    model: nn.Module,
    base_lr: float = 1e-4,
    decay_rate: float = 0.9,
    min_lr: float = 1e-7
) -> List[Dict]:
    """Create parameter groups with layer-wise learning rate decay."""
    param_groups = []
    head_params = []
    fusion_params = []
    concept_params = []
    clinical_backbone_params = {}
    dermoscopic_backbone_params = {}
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'classifier' in name or 'aux_' in name:
            head_params.append(param)
        elif 'tmct_blocks' in name or 'global_pool' in name:
            fusion_params.append(param)
        elif 'concept_embedding' in name:
            concept_params.append(param)
        elif 'metadata_encoder' in name:
            head_params.append(param)
        elif 'backbone.clinical_encoder' in name:
            layer_num = _extract_layer_num(name)
            if layer_num not in clinical_backbone_params:
                clinical_backbone_params[layer_num] = []
            clinical_backbone_params[layer_num].append(param)
        elif 'backbone.dermoscopic_encoder' in name:
            layer_num = _extract_layer_num(name)
            if layer_num not in dermoscopic_backbone_params:
                dermoscopic_backbone_params[layer_num] = []
            dermoscopic_backbone_params[layer_num].append(param)
        else:
            other_params.append(param)
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'name': 'head'
        })
    
    if fusion_params:
        param_groups.append({
            'params': fusion_params,
            'lr': base_lr * 0.9,
            'name': 'fusion'
        })
    
    if concept_params:
        param_groups.append({
            'params': concept_params,
            'lr': base_lr * 0.8,
            'name': 'concept'
        })
    
    for backbone_params, backbone_name in [
        (clinical_backbone_params, 'clinical'),
        (dermoscopic_backbone_params, 'dermoscopic')
    ]:
        if backbone_params:
            max_layer = max(backbone_params.keys())
            for layer_num in sorted(backbone_params.keys(), reverse=True):
                depth = max_layer - layer_num + 1
                layer_lr = max(base_lr * (decay_rate ** (depth + 2)), min_lr)
                param_groups.append({
                    'params': backbone_params[layer_num],
                    'lr': layer_lr,
                    'name': f'{backbone_name}_layer_{layer_num}'
                })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr * 0.5,
            'name': 'other'
        })
    
    return param_groups


def _extract_layer_num(name: str) -> int:
    if 'resblocks' in name:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p == 'resblocks':
                return int(parts[i + 1])
    elif 'blocks' in name:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p == 'blocks':
                return int(parts[i + 1])
    return -1


def create_panderm_model(
    model_name: str = None,
    embed_dim: int = None,
    num_heads: int = None,
    num_classes: int = None,
    dropout: float = None,
    freeze_clinical: int = None,
    freeze_dermoscopic: int = None,
    num_concept_tokens: int = None,
    concept_hidden_dim: int = None,
    tmct_num_layers: int = None,
    use_auxiliary_heads: bool = None,
    pretrained: bool = True,
    **kwargs
) -> TriModalPanDermFusion:
    """Factory function to create PanDerm model with config defaults."""
    config = MODEL_CONFIG_PANDERM
    
    model = TriModalPanDermFusion(
        model_name=model_name or config['model_name'],
        embed_dim=embed_dim or config['embed_dim'],
        num_heads=num_heads or config['num_heads'],
        num_classes=num_classes or config['num_classes'],
        num_concept_tokens=num_concept_tokens or config['num_concept_tokens'],
        concept_hidden_dim=concept_hidden_dim or config.get('concept_hidden_dim', 256),
        tmct_num_layers=tmct_num_layers or config.get('tmct_num_layers', 2),
        dropout=dropout or config['dropout'],
        freeze_clinical=freeze_clinical or config['freeze_clinical'],
        freeze_dermoscopic=freeze_dermoscopic or config['freeze_dermoscopic'],
        use_auxiliary_heads=use_auxiliary_heads if use_auxiliary_heads is not None else config['use_auxiliary_heads'],
        pretrained=pretrained,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Tri-Modal PanDerm Fusion Model")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test with dummy data
    batch_size = 2
    img_size = 224
    num_concepts = 7
    metadata_dim = 11
    
    # Create dummy inputs
    clinical_img = torch.randn(batch_size, 3, img_size, img_size).to(device)
    dermoscopic_img = torch.randn(batch_size, 3, img_size, img_size).to(device)
    monet_scores = torch.rand(batch_size, num_concepts).to(device)
    metadata = torch.randn(batch_size, metadata_dim).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Clinical image: {clinical_img.shape}")
    print(f"  Dermoscopic image: {dermoscopic_img.shape}")
    print(f"  MONET scores: {monet_scores.shape}")
    print(f"  Metadata: {metadata.shape}")
    
    # Create model (will use timm fallback if DermLIP not available)
    print("\nCreating model...")
    model = create_panderm_model(pretrained=True).to(device)
    
    # Test training mode (with auxiliary heads)
    print("\n--- Training Mode ---")
    model.train()
    with torch.no_grad():
        outputs = model(clinical_img, dermoscopic_img, monet_scores, metadata)
    
    if isinstance(outputs, dict):
        print(f"Output (dict):")
        print(f"  Main logits: {outputs['main'].shape}")
        print(f"  Aux clinical: {outputs['aux_clinical'].shape}")
        print(f"  Aux dermoscopic: {outputs['aux_dermoscopic'].shape}")
    else:
        print(f"Output: {outputs.shape}")
    
    # Test eval mode
    print("\n--- Eval Mode ---")
    model.eval()
    with torch.no_grad():
        outputs = model(clinical_img, dermoscopic_img, monet_scores, metadata)
    print(f"Output: {outputs.shape}")
    
    # Test feature extraction
    print("\n--- Feature Extraction ---")
    with torch.no_grad():
        features = model.get_features(clinical_img, dermoscopic_img, monet_scores, metadata)
    print(f"Features: {features.shape}")
    
    # Count parameters
    print("\n--- Model Statistics ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Test layer-wise LR params
    print("\n--- Layer-wise Learning Rates ---")
    lr_params = get_layer_wise_lr_params(model, base_lr=1e-4, decay_rate=0.9)
    for group in lr_params[:5]:  # Show first 5
        print(f"  {group['name']}: lr={group['lr']:.2e}, params={len(group['params'])}")
    print(f"  ... ({len(lr_params)} total groups)")
    
    # Memory estimation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = model(clinical_img, dermoscopic_img, monet_scores, metadata)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"\nPeak GPU memory (batch={batch_size}): {peak_memory:.2f} GB")
    
    print("\n" + "=" * 70)
    print("✅ Model test completed successfully!")
    print("=" * 70)

