"""
Dataset and DataLoader for MILK10k
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(str(Path(__file__).parent))

from config import *


class MILK10kDataset(Dataset):
    """MILK10k Dataset for multi-label skin lesion classification"""
    
    def __init__(self, df, transform=None, fusion_strategy='early', use_metadata=True):
        """
        Args:
            df: DataFrame with image paths and labels
            transform: Albumentations transform
            fusion_strategy: 'early', 'late', or 'feature'
            use_metadata: Whether to include clinical metadata
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fusion_strategy = fusion_strategy
        self.use_metadata = use_metadata
        
        # Prepare metadata columns
        if self.use_metadata:
            self.metadata_cols = CLINICAL_FEATURES + [
                col for col in df.columns if col.startswith('clinical_MONET_') 
                or col.startswith('dermoscopic_MONET_')
            ]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        clinical_img = self._load_image(row['clinical_image_path'])
        dermoscopic_img = self._load_image(row['dermoscopic_image_path'])
        
        # Apply transforms
        if self.transform:
            clinical_img = self.transform(image=clinical_img)['image']
            dermoscopic_img = self.transform(image=dermoscopic_img)['image']
        
        # Fusion strategy
        if self.fusion_strategy == 'early':
            # Concatenate images along channel dimension
            image = torch.cat([clinical_img, dermoscopic_img], dim=0)  # [6, H, W]
        elif self.fusion_strategy == 'late':
            # Return both images separately for late fusion
            image = (clinical_img, dermoscopic_img)
        else:
            # For feature-level fusion, return both images
            image = (clinical_img, dermoscopic_img)
        
        # Get labels
        labels = torch.tensor(
            row[DIAGNOSIS_CATEGORIES].values.astype(np.float32)
        )
        
        # Get metadata if required
        if self.use_metadata:
            metadata = self._process_metadata(row)
            return image, labels, metadata
        else:
            return image, labels
    
    def _load_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _process_metadata(self, row):
        """Process clinical metadata"""
        metadata_dict = {}
        
        # Process categorical features
        if 'sex' in self.metadata_cols:
            sex_map = {'male': 0, 'female': 1}
            metadata_dict['sex'] = sex_map.get(row['sex'], -1)
        
        if 'site' in self.metadata_cols:
            site_map = {
                'head_neck_face': 0,
                'upper_extremity': 1,
                'lower_extremity': 2,
                'trunk': 3,
                'palms_soles': 4,
                'oral_genital': 5
            }
            metadata_dict['site'] = site_map.get(row['site'], -1)
        
        # Process numerical features
        if 'age_approx' in self.metadata_cols:
            metadata_dict['age'] = row['age_approx'] / 100.0  # Normalize
        
        if 'skin_tone_class' in self.metadata_cols:
            metadata_dict['skin_tone'] = row['skin_tone_class'] / 5.0  # Normalize
        
        # Process MONET scores (already normalized 0-1)
        for col in self.metadata_cols:
            if col.startswith('clinical_MONET_') or col.startswith('dermoscopic_MONET_'):
                metadata_dict[col] = row[col]
        
        # Convert to tensor
        metadata_tensor = torch.tensor(
            list(metadata_dict.values()), dtype=torch.float32
        )
        
        return metadata_tensor


def get_transforms(image_size=384, augmentation=True):
    """Get image transforms for training and validation"""
    
    if augmentation:
        # Training transforms with augmentation
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size//8,
                max_width=image_size//8,
                p=0.3
            ),
            A.Normalize(
                mean=IMAGE_CONFIG['normalize_mean'],
                std=IMAGE_CONFIG['normalize_std']
            ),
            ToTensorV2()
        ])
    else:
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=IMAGE_CONFIG['normalize_mean'],
                std=IMAGE_CONFIG['normalize_std']
            ),
            ToTensorV2()
        ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=IMAGE_CONFIG['normalize_mean'],
            std=IMAGE_CONFIG['normalize_std']
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def get_dataloaders(train_df, val_df, batch_size=16, num_workers=4, 
                    image_size=384, fusion_strategy='early', use_metadata=True):
    """Create DataLoaders for training and validation"""
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=image_size,
        augmentation=AUGMENTATION_CONFIG['use_augmentation']
    )
    
    # Create datasets
    train_dataset = MILK10kDataset(
        train_df,
        transform=train_transform,
        fusion_strategy=fusion_strategy,
        use_metadata=use_metadata
    )
    
    val_dataset = MILK10kDataset(
        val_df,
        transform=val_transform,
        fusion_strategy=fusion_strategy,
        use_metadata=use_metadata
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=DEVICE_CONFIG['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE_CONFIG['pin_memory']
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    train_df = pd.read_csv(PREPROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PREPROCESSED_DIR / "val_data.csv")
    
    train_loader, val_loader = get_dataloaders(
        train_df, val_df,
        batch_size=4,
        num_workers=0
    )
    
    # Test one batch
    print("\nTesting one batch...")
    for batch in train_loader:
        if MODEL_CONFIG['use_metadata']:
            images, labels, metadata = batch
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Metadata shape: {metadata.shape}")
        else:
            images, labels = batch
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
        break
    
    print("\nDataset loading test completed!")
