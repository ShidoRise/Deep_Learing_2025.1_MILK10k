"""
Configuration file for MILK10k Skin Lesion Classification Project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset paths
TRAIN_INPUT_DIR = DATASET_DIR / "MILK10k_Training_Input"
TEST_INPUT_DIR = DATASET_DIR / "MILK10k_Test_Input"
TRAIN_GT_FILE = DATASET_DIR / "MILK10k_Training_GroundTruth.csv"
TRAIN_METADATA_FILE = DATASET_DIR / "MILK10k_Training_Metadata.csv"
TRAIN_SUPPLEMENT_FILE = DATASET_DIR / "MILK10k_Training_Supplement.csv"

# Diagnostic categories
DIAGNOSIS_CATEGORIES = [
    'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF',
    'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
]

# Image types
IMAGE_TYPES = ['clinical: close-up', 'dermoscopic']

# MONET features (7 features from supplement file)
MONET_FEATURES = [
    'MONET_ulceration_crust',
    'MONET_hair',
    'MONET_vasculature_vessels',
    'MONET_erythema',
    'MONET_pigmented',
    'MONET_gel_water_drop_fluid_dermoscopy_liquid',
    'MONET_skin_markings_pen_ink_purple_pen'
]

# Clinical metadata features
CLINICAL_FEATURES = [
    'age_approx',
    'sex',
    'skin_tone_class',
    'site'
]

# =============================================================================
# BASELINE MODEL CONFIG (EfficientNet-B3)
# =============================================================================
MODEL_CONFIG = {
    'architecture': 'efficientnet_b3',
    'pretrained': True,
    'num_classes': 11,
    'dropout': 0.3,
    'use_metadata': True,
    'metadata_dim': 18,
    'use_attention': True,  # Use attention fusion (dual backbone with learned weights)
}

IMAGE_CONFIG = {
    'image_size': 384,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'fusion_strategy': 'late',  # 'early' for 6-channel, 'late' for separate backbones
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'min_lr': 1e-6,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,
    'mixed_precision': True,
    'random_seed': 42,
    'num_workers': 4,
    'save_every': 5,
    'checkpoint_dir': str(MODELS_DIR),
    'log_dir': str(LOGS_DIR),
}

LOSS_CONFIG = {
    'type': 'bce_with_logits',
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'use_class_weights': True,
}

# =============================================================================
# PANDERM MODEL CONFIG (Tri-Modal PanDerm Fusion)
# =============================================================================
MODEL_CONFIG_PANDERM = {
    # Backbone
    'model_name': 'redlessone/DermLIP_PanDerm-base-w-PubMed-256',
    'embed_dim': 768,
    'num_heads': 8,
    'num_classes': 11,
    'dropout': 0.1,
    
    # Layer freezing for transfer learning
    'freeze_clinical': 6,      # Freeze first 6 layers of clinical encoder
    'freeze_dermoscopic': 4,   # Freeze first 4 layers of dermoscopic encoder
    
    # MONET concept tokens
    'num_concept_tokens': 7,  # One token per MONET feature (7 features)
    'concept_hidden_dim': 256,
    
    # Fusion
    'tmct_num_layers': 2,      # Number of TMCT fusion blocks
    'mlp_ratio': 4.0,
    
    # Auxiliary heads for deep supervision
    'use_auxiliary_heads': True,
    'aux_loss_weight': 0.3,
}

IMAGE_CONFIG_PANDERM = {
    'image_size': 224,  # ViT native resolution
    'normalize_mean': [0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
    'normalize_std': [0.26862954, 0.26130258, 0.27577711],
}

TRAIN_CONFIG_PANDERM = {
    # Batch and epochs
    'batch_size': 32,          # A100: 96, RTX 4090: 32, RTX 3090: 16
    'num_epochs': 60,
    'gradient_accumulation': 2,  # Effective batch = 64
    
    # Learning rate with layer-wise decay
    'base_lr': 1e-4,           # For head/fusion layers
    'backbone_lr_decay': 0.9,  # Decay rate per layer
    'min_lr': 1e-7,
    'weight_decay': 0.05,      # Higher for large ViT
    
    # Scheduler
    'scheduler': 'cosine_warmup',
    'warmup_epochs': 3,
    
    # Training stability
    'early_stopping_patience': 12,
    'gradient_clip': 1.0,
    'mixed_precision': 'bf16',   # 'bf16' for A100, 'fp16' for consumer GPUs
    
    # Dropout strategies for robustness
    'modality_dropout': 0.2,   # Probability of zeroing clinical image
    'concept_dropout': 0.1,    # Probability of zeroing individual MONET scores
    
    # System
    'random_seed': 42,
    'num_workers': 8,
    'save_every': 5,
    'checkpoint_dir': str(MODELS_DIR),
    'log_dir': str(LOGS_DIR),
    
    # EMA for better generalization
    'use_ema': True,
    'ema_decay': 0.9999,
}

LOSS_CONFIG_PANDERM = {
    # Compound loss: Focal + Soft F1
    'type': 'compound',
    'focal_gamma': 2.0,
    'focal_weight': 0.5,
    'soft_f1_weight': 0.5,
    'use_class_weights': True,
    
    # Auxiliary loss
    'aux_loss_weight': 0.3,
}

# =============================================================================
# XGBOOST STACKING CONFIG
# =============================================================================
XGBOOST_CONFIG = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda',
}

# =============================================================================
# ENSEMBLE CONFIG
# =============================================================================
ENSEMBLE_CONFIG = {
    # Initial weights (will be optimized)
    'panderm_weight': 0.4,
    'efficientnet_weight': 0.3,
    'xgboost_weight': 0.3,
    
    # TTA settings
    'use_tta': True,
    'tta_augments': 8,  # 4 rotations x 2 flips
}

# =============================================================================
# DATA CONFIG
# =============================================================================
DATA_SPLIT = {
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'random_seed': 42,
    'stratify': True,
}

AUGMENTATION_CONFIG = {
    'use_augmentation': True,
    'random_rotation': 20,
    'random_flip': True,
    'random_scale': (0.9, 1.1),
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'cutout': {
        'use': True,
        'num_holes': 3,
        'max_h_size': 30,
        'max_w_size': 30
    },
    'mixup': {
        'use': False,
        'alpha': 0.2
    }
}

# =============================================================================
# EVALUATION CONFIG
# =============================================================================
EVAL_CONFIG = {
    'threshold': 0.5,
    'metrics': ['macro_f1', 'micro_f1', 'per_class_f1', 'auc_roc'],
    'save_predictions': True,
    'save_visualizations': True,
}

# =============================================================================
# HARDWARE CONFIG
# =============================================================================
DEVICE_CONFIG = {
    'use_cuda': True,
    'num_workers': 4,
    'pin_memory': True,
}

# =============================================================================
# LOGGING CONFIG
# =============================================================================
LOGGING_CONFIG = {
    'log_interval': 10,
    'save_interval': 5,
    'use_tensorboard': True,
    'use_wandb': False,
    'wandb_project': 'milk10k-skin-lesion',
}
