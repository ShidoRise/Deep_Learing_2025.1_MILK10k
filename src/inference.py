"""
Inference script for MILK10k skin lesion classification
Generates predictions for test set
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from utils import get_device, load_checkpoint
from dataset import MILK10kDataset, get_transforms
from models import create_model


class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = Path(model_path)
        
        print("Loading model...")
        self.model = self._load_model()
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
    
    def _load_model(self):
        model = create_model(
            architecture=MODEL_CONFIG['architecture'],
            num_classes=len(DIAGNOSIS_CATEGORIES),
            pretrained=False,  # Load our trained weights
            fusion_strategy=IMAGE_CONFIG['fusion_strategy'],
            use_attention=MODEL_CONFIG.get('use_attention', False),
            use_metadata=MODEL_CONFIG['use_metadata'],
            metadata_dim=MODEL_CONFIG['metadata_dim'],
            dropout=MODEL_CONFIG['dropout']
        )
        
        checkpoint = load_checkpoint(model, None, self.model_path, device=self.device)
        
        return model.to(self.device)
    
    def predict(self, test_df, batch_size=32, num_workers=4):
        _, test_transform = get_transforms(
            image_size=IMAGE_CONFIG['image_size'],
            augment=False
        )
        
        has_image_paths = 'clinical_image_path' in test_df.columns and 'dermoscopic_image_path' in test_df.columns
        image_dir = None if has_image_paths else TEST_INPUT_DIR
        
        test_dataset = MILK10kDataset(
            df=test_df,
            image_dir=image_dir,
            transform=test_transform,
            fusion_strategy=IMAGE_CONFIG['fusion_strategy'],
            use_metadata=MODEL_CONFIG['use_metadata'],
            is_test=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"\nTest samples: {len(test_dataset):,}")
        print(f"Test batches: {len(test_loader)}")
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Predicting'):
                if len(batch) == 2:
                    images, metadata = batch
                    metadata = metadata.to(self.device)
                else:
                    images = batch[0]
                    metadata = None
                
                # Handle both early fusion (single tensor) and late fusion (tuple)
                if isinstance(images, (list, tuple)):
                    images = (images[0].to(self.device), images[1].to(self.device))
                else:
                    images = images.to(self.device)
                
                if metadata is not None:
                    outputs = self.model(images, metadata)
                else:
                    outputs = self.model(images)
                
                probs = torch.sigmoid(outputs)
                all_predictions.append(probs.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        
        print(f"\nPredictions shape: {predictions.shape}")
        
        return predictions
    
    def predict_with_tta(self, test_df, batch_size=32, num_workers=4, n_tta=5):
        print(f"\nUsing Test Time Augmentation (TTA) with {n_tta} iterations")
        
        has_image_paths = 'clinical_image_path' in test_df.columns and 'dermoscopic_image_path' in test_df.columns
        image_dir = None if has_image_paths else TEST_INPUT_DIR
        
        all_tta_predictions = []
        
        for tta_idx in range(n_tta):
            print(f"\nTTA iteration {tta_idx + 1}/{n_tta}")
            
            tta_transform, _ = get_transforms(
                image_size=IMAGE_CONFIG['image_size'],
                augment=True
            )
            
            test_dataset = MILK10kDataset(
                df=test_df,
                image_dir=image_dir,
                transform=tta_transform,
                fusion_strategy=IMAGE_CONFIG['fusion_strategy'],
                use_metadata=MODEL_CONFIG['use_metadata'],
                is_test=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            tta_predictions = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'TTA {tta_idx+1}'):
                    if len(batch) == 2:
                        images, metadata = batch
                        metadata = metadata.to(self.device)
                    else:
                        images = batch[0]
                        metadata = None
                    
                    # Handle both early fusion (single tensor) and late fusion (tuple)
                    if isinstance(images, (list, tuple)):
                        images = (images[0].to(self.device), images[1].to(self.device))
                    else:
                        images = images.to(self.device)
                    
                    if metadata is not None:
                        outputs = self.model(images, metadata)
                    else:
                        outputs = self.model(images)
                    
                    probs = torch.sigmoid(outputs)
                    tta_predictions.append(probs.cpu().numpy())
            
            tta_predictions = np.vstack(tta_predictions)
            all_tta_predictions.append(tta_predictions)
        
        predictions = np.mean(all_tta_predictions, axis=0)
        
        print(f"\nFinal predictions shape: {predictions.shape}")
        
        return predictions


def create_submission(predictions, test_df, output_path):
    """
    Create submission file with probability values.
    
    Note: ISIC MILK10k challenge expects floating-point probabilities in [0.0, 1.0].
    The challenge system will apply threshold >= 0.5 during evaluation.
    """
    submission = pd.DataFrame({
        'lesion_id': test_df['lesion_id'].values
    })
    
    # Save raw probabilities (NOT binary) as required by ISIC challenge
    for i, category in enumerate(DIAGNOSIS_CATEGORIES):
        submission[category] = predictions[:, i]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print("\nSubmission preview (probabilities):")
    print(submission.head())
    
    # Show statistics based on 0.5 threshold (for reference only)
    print("\nPrediction statistics (threshold >= 0.5):")
    for category in DIAGNOSIS_CATEGORIES:
        count = (submission[category] >= 0.5).sum()
        mean_prob = submission[category].mean()
        print(f"  {category}: {count:,} positive, mean prob: {mean_prob:.4f}")
    
    return submission


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MILK10k Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Path to save submission file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation')
    parser.add_argument('--n_tta', type=int, default=5,
                        help='Number of TTA iterations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    
    args = parser.parse_args()
    
    device = get_device()
    
    print("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(test_df):,}")
    
    predictor = Predictor(
        model_path=args.model_path,
        device=device
    )
    
    if args.use_tta:
        predictions = predictor.predict_with_tta(
            test_df=test_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_tta=args.n_tta
        )
    else:
        predictions = predictor.predict(
            test_df=test_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    submission = create_submission(
        predictions=predictions,
        test_df=test_df,
        output_path=args.output_path,
        threshold=args.threshold
    )
    
    print("\n✅ Inference completed successfully!")


if __name__ == "__main__":
    main()
