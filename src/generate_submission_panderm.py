"""
Ensemble submission generator for PanDerm model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import sys

sys.path.append(str(Path(__file__).parent))

from config import *
from utils import get_device, load_checkpoint
from models_panderm import create_panderm_model
from train_panderm import PanDermDataset, get_panderm_transforms
from xgboost_stacking import EnsemblePredictor, load_xgboost, load_ensemble_config, predict_xgboost


def prepare_test_data():
    test_metadata_path = DATASET_DIR / 'MILK10k_Test_Metadata.csv'
    
    if not test_metadata_path.exists():
        test_dirs = sorted([d for d in TEST_INPUT_DIR.iterdir() if d.is_dir()])
        test_data = []
        for test_dir in test_dirs:
            lesion_id = test_dir.name
            if not lesion_id.startswith('IL_'):
                continue
            image_files = sorted(list(test_dir.glob('*.jpg')))
            if len(image_files) < 2:
                continue
            test_data.append({
                'lesion_id': lesion_id,
                'clinical_image_path': str(image_files[0]),
                'dermoscopic_image_path': str(image_files[1]),
                'age_approx': 50.0, 'sex': 'unknown', 'site': 'unknown', 'skin_tone_class': 3.0
            })
        test_df = pd.DataFrame(test_data)
    else:
        test_metadata = pd.read_csv(test_metadata_path)
        lesion_data = []
        for lesion_id, group in test_metadata.groupby('lesion_id'):
            clinical_row = group[group['image_type'] == 'clinical: close-up']
            dermoscopic_row = group[group['image_type'] == 'dermoscopic']
            if len(clinical_row) == 0 or len(dermoscopic_row) == 0:
                continue
            clinical_row = clinical_row.iloc[0]
            dermoscopic_row = dermoscopic_row.iloc[0]
            
            lesion_entry = {
                'lesion_id': lesion_id,
                'clinical_image_path': str(TEST_INPUT_DIR / lesion_id / f"{clinical_row['isic_id']}.jpg"),
                'dermoscopic_image_path': str(TEST_INPUT_DIR / lesion_id / f"{dermoscopic_row['isic_id']}.jpg"),
                'age_approx': clinical_row.get('age_approx', 50.0),
                'sex': clinical_row.get('sex', 'unknown'),
                'site': clinical_row.get('site', 'unknown'),
                'skin_tone_class': clinical_row.get('skin_tone_class', 3.0),
            }
            
            for col in [c for c in clinical_row.index if c.startswith('MONET_')]:
                lesion_entry[f'clinical_{col}'] = clinical_row[col]
                lesion_entry[f'dermoscopic_{col}'] = dermoscopic_row[col]
            
            lesion_data.append(lesion_entry)
        
        test_df = pd.DataFrame(lesion_data)
        test_df['age_approx'] = test_df['age_approx'].fillna(50.0)
        test_df['sex'] = test_df['sex'].fillna('unknown')
        test_df['site'] = test_df['site'].fillna('unknown')
        test_df['skin_tone_class'] = test_df['skin_tone_class'].fillna(3.0)
    
    output_path = PREPROCESSED_DIR / 'test_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    print(f"Test data: {len(test_df)} samples")
    return test_df


class PanDermTestDataset(PanDermDataset):
    def __init__(self, df, transform=None):
        super().__init__(df, transform=transform, is_test=True)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_img = self._load_image(row['clinical_image_path'])
        dermoscopic_img = self._load_image(row['dermoscopic_image_path'])
        
        if self.transform:
            clinical_img = self.transform(image=clinical_img)['image']
            dermoscopic_img = self.transform(image=dermoscopic_img)['image']
        
        monet_scores = self._get_monet_scores(row)
        metadata = self._get_metadata(row)
        lesion_id = row['lesion_id']
        
        return clinical_img, dermoscopic_img, monet_scores, metadata, lesion_id


def predict_panderm_only(model, dataloader, device='cuda', use_tta=False):
    model.eval()
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            clinical_img, dermoscopic_img, monet_scores, metadata, lesion_ids = batch
            clinical_img = clinical_img.to(device)
            dermoscopic_img = dermoscopic_img.to(device)
            monet_scores = monet_scores.to(device)
            metadata = metadata.to(device)
            
            if use_tta:
                probs = predict_with_tta(model, clinical_img, dermoscopic_img, monet_scores, metadata)
            else:
                with torch.amp.autocast('cuda'):
                    outputs = model(clinical_img, dermoscopic_img, monet_scores, metadata)
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs
                probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_ids.extend(lesion_ids)
    
    return np.vstack(all_probs), all_ids


def predict_with_tta(model, clinical_img, dermoscopic_img, monet_scores, metadata, num_augs=8):
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
    
    all_preds = []
    for aug in augments[:num_augs]:
        aug_clinical = aug(clinical_img)
        aug_dermoscopic = aug(dermoscopic_img)
        
        with torch.amp.autocast('cuda'):
            outputs = model(aug_clinical, aug_dermoscopic, monet_scores, metadata)
        
        if isinstance(outputs, dict):
            logits = outputs['main']
        else:
            logits = outputs
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
    
    return np.mean(all_preds, axis=0)


def predict_ensemble(panderm_model, xgb_model, dataloader, weights, device='cuda', use_tta=False):
    panderm_model.eval()
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Ensemble predicting'):
            clinical_img, dermoscopic_img, monet_scores, metadata, lesion_ids = batch
            clinical_img = clinical_img.to(device)
            dermoscopic_img = dermoscopic_img.to(device)
            monet_scores = monet_scores.to(device)
            metadata = metadata.to(device)
            
            with torch.amp.autocast('cuda'):
                panderm_out = panderm_model(clinical_img, dermoscopic_img, monet_scores, metadata)
                features = panderm_model.get_features(clinical_img, dermoscopic_img, monet_scores, metadata)
            
            if isinstance(panderm_out, dict):
                panderm_logits = panderm_out['main']
            else:
                panderm_logits = panderm_out
            panderm_probs = torch.sigmoid(panderm_logits).cpu().numpy()
            
            xgb_input = np.concatenate([
                features.cpu().numpy(),
                metadata.cpu().numpy(),
                monet_scores.cpu().numpy(),
                panderm_probs
            ], axis=1)
            xgb_probs = predict_xgboost(xgb_model, xgb_input)
            
            ensemble_probs = weights[0] * panderm_probs + weights[1] * xgb_probs
            
            all_probs.append(ensemble_probs)
            all_ids.extend(lesion_ids)
    
    return np.vstack(all_probs), all_ids


def create_submission(predictions, lesion_ids, output_path, threshold=0.5):
    submission_data = {'lesion_id': lesion_ids}
    
    # Keep probabilities as floating point values (not binary)
    for i, cat in enumerate(DIAGNOSIS_CATEGORIES):
        submission_data[cat] = predictions[:, i]
    
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df.sort_values('lesion_id').reset_index(drop=True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved: {output_path}")
    print(f"Shape: {submission_df.shape}")
    print(f"\nPrediction statistics (mean probability per class):")
    for cat in DIAGNOSIS_CATEGORIES:
        print(f"  {cat}: mean={submission_df[cat].mean():.4f}, std={submission_df[cat].std():.4f}")
    
    return submission_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panderm_path', type=str, default=None)
    parser.add_argument('--xgb_path', type=str, default=None)
    parser.add_argument('--weights_path', type=str, default=None)
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--use_ensemble', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    device = get_device()
    
    test_df = prepare_test_data()
    
    transform = get_panderm_transforms(IMAGE_CONFIG_PANDERM['image_size'], augment=False)
    test_dataset = PanDermTestDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Loading PanDerm model...")
    panderm = create_panderm_model(pretrained=False)
    panderm_path = args.panderm_path or (Path(MODELS_DIR) / 'panderm_best.pth')
    if Path(panderm_path).exists():
        load_checkpoint(panderm, filepath=panderm_path, device=device)
        print(f"Loaded: {panderm_path}")
    else:
        print(f"Warning: {panderm_path} not found, using random weights")
    panderm = panderm.to(device).eval()
    
    if args.use_ensemble:
        xgb_path = args.xgb_path or (Path(MODELS_DIR) / 'xgboost_stacking.pkl')
        weights_path = args.weights_path or (Path(MODELS_DIR) / 'ensemble_weights.json')
        
        if Path(xgb_path).exists() and Path(weights_path).exists():
            print("Loading XGBoost and ensemble weights...")
            xgb_model = load_xgboost(xgb_path)
            weights = load_ensemble_config(weights_path)
            print(f"Ensemble weights: {weights}")
            
            predictions, lesion_ids = predict_ensemble(panderm, xgb_model, test_loader, weights, device)
            output_name = 'submission_ensemble.csv'
        else:
            print("XGBoost or weights not found, using PanDerm only")
            predictions, lesion_ids = predict_panderm_only(panderm, test_loader, device, args.use_tta)
            output_name = 'submission_panderm_tta.csv' if args.use_tta else 'submission_panderm.csv'
    else:
        predictions, lesion_ids = predict_panderm_only(panderm, test_loader, device, args.use_tta)
        output_name = 'submission_panderm_tta.csv' if args.use_tta else 'submission_panderm.csv'
    
    output_path = args.output or (RESULTS_DIR / output_name)
    submission = create_submission(predictions, lesion_ids, output_path)
    
    print("\nSubmission generated!")
    return submission


if __name__ == "__main__":
    main()

