"""
Model architectures for MILK10k skin lesion classification
"""
import torch
import torch.nn as nn
import timm
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import *


class SkinLesionClassifier(nn.Module):
    def __init__(
        self,
        architecture='efficientnet_b3',
        num_classes=11,
        pretrained=True,
        fusion_strategy='early',
        use_metadata=True,
        metadata_dim=18,
        dropout=0.3
    ):
        super(SkinLesionClassifier, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        self.use_metadata = use_metadata
        self.num_classes = num_classes
        
        if fusion_strategy == 'early':
            self.backbone = timm.create_model(
                architecture,
                pretrained=pretrained,
                in_chans=6,
                num_classes=0,  # Remove classification head
                global_pool='avg'
            )
        else:
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
        
        if fusion_strategy == 'early':
            feature_dim = self.backbone.num_features
        else:
            feature_dim = self.clinical_backbone.num_features * 2
        
        if use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            )
            feature_dim += 64
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images, metadata=None):
        if self.fusion_strategy == 'early':
            features = self.backbone(images)
        else:
            clinical_img, dermoscopic_img = images
            clinical_features = self.clinical_backbone(clinical_img)
            dermoscopic_features = self.dermoscopic_backbone(dermoscopic_img)
            features = torch.cat([clinical_features, dermoscopic_features], dim=1)
        
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            features = torch.cat([features, metadata_features], dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, clinical_features, dermoscopic_features):
        combined = torch.cat([clinical_features, dermoscopic_features], dim=1)
        attention_weights = self.attention(combined)
        clinical_weight = attention_weights[:, 0:1]
        dermoscopic_weight = attention_weights[:, 1:2]
        
        fused_features = (
            clinical_weight * clinical_features +
            dermoscopic_weight * dermoscopic_features
        )
        
        return fused_features


class SkinLesionClassifierWithAttention(nn.Module):
    def __init__(
        self,
        architecture='efficientnet_b3',
        num_classes=11,
        pretrained=True,
        use_metadata=True,
        metadata_dim=18,
        dropout=0.3
    ):
        super(SkinLesionClassifierWithAttention, self).__init__()
        
        self.use_metadata = use_metadata
        
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
        
        feature_dim = self.clinical_backbone.num_features
        
        self.attention_fusion = AttentionFusion(feature_dim)
        
        if use_metadata:
            self.metadata_fc = nn.Sequential(
                nn.Linear(metadata_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            )
            feature_dim += 64
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images, metadata=None):
        clinical_img, dermoscopic_img = images
        
        clinical_features = self.clinical_backbone(clinical_img)
        dermoscopic_features = self.dermoscopic_backbone(dermoscopic_img)
        
        features = self.attention_fusion(clinical_features, dermoscopic_features)
        
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_fc(metadata)
            features = torch.cat([features, metadata_features], dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def create_model(
    architecture='efficientnet_b3',
    num_classes=11,
    pretrained=True,
    fusion_strategy='early',
    use_attention=False,
    use_metadata=True,
    metadata_dim=18,
    dropout=0.3
):
    if use_attention:
        model = SkinLesionClassifierWithAttention(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            use_metadata=use_metadata,
            metadata_dim=metadata_dim,
            dropout=dropout
        )
    else:
        model = SkinLesionClassifier(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_strategy=fusion_strategy,
            use_metadata=use_metadata,
            metadata_dim=metadata_dim,
            dropout=dropout
        )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test early fusion
    print("\n1. Early Fusion Model:")
    model_early = create_model(
        architecture='efficientnet_b3',
        fusion_strategy='early',
        use_metadata=True
    )
    
    batch_size = 2
    images_early = torch.randn(batch_size, 6, 384, 384)
    metadata = torch.randn(batch_size, 18)
    
    with torch.no_grad():
        output = model_early(images_early, metadata)
    
    print(f"Input shape: {images_early.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test late fusion
    print("\n2. Late Fusion Model:")
    model_late = create_model(
        architecture='efficientnet_b3',
        fusion_strategy='late',
        use_metadata=True
    )
    
    clinical_img = torch.randn(batch_size, 3, 384, 384)
    dermoscopic_img = torch.randn(batch_size, 3, 384, 384)
    
    with torch.no_grad():
        output = model_late((clinical_img, dermoscopic_img), metadata)
    
    print(f"Clinical image shape: {clinical_img.shape}")
    print(f"Dermoscopic image shape: {dermoscopic_img.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test attention fusion
    print("\n3. Attention Fusion Model:")
    model_attention = create_model(
        architecture='efficientnet_b3',
        use_attention=True,
        use_metadata=True
    )
    
    with torch.no_grad():
        output = model_attention((clinical_img, dermoscopic_img), metadata)
    
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    from utils import count_parameters
    print("\n4. Model Parameters:")
    count_parameters(model_early)
    
    print("\nModel creation test completed!")
