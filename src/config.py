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

# MONET features
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

# Model configuration
MODEL_CONFIG = {
    'architecture': 'efficientnet_b3',
    'pretrained': True,
    'num_classes': 11,
    'dropout': 0.3,
    'use_metadata': True,
    'metadata_dim': 18,
}

# Image preprocessing
IMAGE_CONFIG = {
    'image_size': 384,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'fusion_strategy': 'late',
}

# Training configuration
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

# Loss function
LOSS_CONFIG = {
    'type': 'bce_with_logits',
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'use_class_weights': True,
}

# Data split
DATA_SPLIT = {
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'random_seed': 42,
    'stratify': True,
}

# Data augmentation
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

# Evaluation
EVAL_CONFIG = {
    'threshold': 0.5,
    'metrics': ['macro_f1', 'micro_f1', 'per_class_f1', 'auc_roc'],
    'save_predictions': True,
    'save_visualizations': True,
}

# Hardware
DEVICE_CONFIG = {
    'use_cuda': True,
    'num_workers': 4,
    'pin_memory': True,
}

# Logging
LOGGING_CONFIG = {
    'log_interval': 10,
    'save_interval': 5,
    'use_tensorboard': True,
    'use_wandb': False,
    'wandb_project': 'milk10k-skin-lesion',
}
