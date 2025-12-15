"""
XGBoost Hybrid Stacking for PanDerm Ensemble
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import joblib
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
import sys

sys.path.append(str(Path(__file__).parent))

from config import *

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def extract_features(model, dataloader, device='cuda'):
    """Extract features from trained PanDerm model for XGBoost."""
    model.eval()
    
    all_features = []
    all_monet = []
    all_metadata = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            if len(batch) == 5:
                clinical_img, dermoscopic_img, monet_scores, metadata, labels = batch
                has_labels = True
            else:
                clinical_img, dermoscopic_img, monet_scores, metadata = batch
                has_labels = False
                labels = None
            
            clinical_img = clinical_img.to(device)
            dermoscopic_img = dermoscopic_img.to(device)
            monet_scores = monet_scores.to(device)
            metadata = metadata.to(device)
            
            with torch.amp.autocast('cuda'):
                features = model.get_features(clinical_img, dermoscopic_img, monet_scores, metadata)
                outputs = model(clinical_img, dermoscopic_img, monet_scores, metadata)
            
            if isinstance(outputs, dict):
                logits = outputs['main']
            else:
                logits = outputs
            
            all_features.append(features.cpu().numpy())
            all_monet.append(monet_scores.cpu().numpy())
            all_metadata.append(metadata.cpu().numpy())
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            
            if has_labels:
                all_labels.append(labels.numpy())
    
    features = np.vstack(all_features)
    monet = np.vstack(all_monet)
    metadata = np.vstack(all_metadata)
    logits = np.vstack(all_logits)
    
    combined = np.concatenate([features, metadata, monet, logits], axis=1)
    
    if all_labels:
        labels = np.vstack(all_labels)
        return combined, labels
    return combined, None


def train_xgboost(train_features, train_labels, val_features=None, val_labels=None):
    """Train XGBoost classifier for multi-label classification."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed. pip install xgboost")
    
    xgb_params = {
        'max_depth': XGBOOST_CONFIG.get('max_depth', 6),
        'learning_rate': XGBOOST_CONFIG.get('learning_rate', 0.05),
        'n_estimators': XGBOOST_CONFIG.get('n_estimators', 300),
        'min_child_weight': XGBOOST_CONFIG.get('min_child_weight', 3),
        'subsample': XGBOOST_CONFIG.get('subsample', 0.8),
        'colsample_bytree': XGBOOST_CONFIG.get('colsample_bytree', 0.8),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    base_model = xgb.XGBClassifier(**xgb_params)
    multi_model = MultiOutputClassifier(base_model, n_jobs=1)
    
    print("Training XGBoost...")
    multi_model.fit(train_features, train_labels)
    
    if val_features is not None and val_labels is not None:
        val_preds = multi_model.predict(val_features)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"XGBoost Val Macro F1: {val_f1:.4f}")
    
    return multi_model


def predict_xgboost(model, features):
    """Get probability predictions from XGBoost."""
    proba_list = model.predict_proba(features)
    probs = np.stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba_list], axis=1)
    return probs


def optimize_ensemble_weights(panderm_probs, efficientnet_probs, xgboost_probs, true_labels):
    """Find optimal ensemble weights using Nelder-Mead."""
    
    def objective(weights):
        w = np.abs(weights)
        w = w / w.sum()
        
        if efficientnet_probs is not None:
            ensemble = w[0] * panderm_probs + w[1] * efficientnet_probs + w[2] * xgboost_probs
        else:
            ensemble = w[0] * panderm_probs + w[1] * xgboost_probs
        
        preds = (ensemble > 0.5).astype(int)
        macro_f1 = f1_score(true_labels, preds, average='macro')
        return -macro_f1
    
    if efficientnet_probs is not None:
        x0 = [0.4, 0.3, 0.3]
        bounds = [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8)]
    else:
        x0 = [0.6, 0.4]
        bounds = [(0.2, 0.9), (0.1, 0.8)]
    
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
    
    weights = np.abs(result.x)
    weights = weights / weights.sum()
    
    print(f"Optimal weights: {weights}")
    print(f"Best Macro F1: {-result.fun:.4f}")
    
    return weights


class EnsemblePredictor:
    """Ensemble combining PanDerm, EfficientNet (optional), and XGBoost."""
    
    def __init__(self, panderm_model, xgboost_model, efficientnet_model=None, weights=None, device='cuda'):
        self.panderm = panderm_model.to(device).eval()
        self.xgboost = xgboost_model
        self.efficientnet = efficientnet_model.to(device).eval() if efficientnet_model else None
        self.device = device
        
        if weights is not None:
            self.weights = weights
        elif self.efficientnet is not None:
            self.weights = np.array([0.4, 0.3, 0.3])
        else:
            self.weights = np.array([0.6, 0.4])
    
    @torch.no_grad()
    def predict(self, clinical_img, dermoscopic_img, monet_scores, metadata):
        clinical_img = clinical_img.to(self.device)
        dermoscopic_img = dermoscopic_img.to(self.device)
        monet_scores = monet_scores.to(self.device)
        metadata = metadata.to(self.device)
        
        with torch.amp.autocast('cuda'):
            panderm_out = self.panderm(clinical_img, dermoscopic_img, monet_scores, metadata)
            panderm_features = self.panderm.get_features(clinical_img, dermoscopic_img, monet_scores, metadata)
        
        if isinstance(panderm_out, dict):
            panderm_logits = panderm_out['main']
        else:
            panderm_logits = panderm_out
        panderm_probs = torch.sigmoid(panderm_logits).cpu().numpy()
        
        xgb_input = np.concatenate([
            panderm_features.cpu().numpy(),
            metadata.cpu().numpy(),
            monet_scores.cpu().numpy(),
            panderm_probs
        ], axis=1)
        xgb_probs = predict_xgboost(self.xgboost, xgb_input)
        
        if self.efficientnet is not None:
            with torch.amp.autocast('cuda'):
                eff_out = self.efficientnet((clinical_img, dermoscopic_img), metadata)
            eff_probs = torch.sigmoid(eff_out).cpu().numpy()
            ensemble = self.weights[0] * panderm_probs + self.weights[1] * eff_probs + self.weights[2] * xgb_probs
        else:
            ensemble = self.weights[0] * panderm_probs + self.weights[1] * xgb_probs
        
        return ensemble
    
    def predict_with_tta(self, clinical_img, dermoscopic_img, monet_scores, metadata, num_augs=8):
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
            pred = self.predict(aug_clinical, aug_dermoscopic, monet_scores, metadata)
            all_preds.append(pred)
        
        return np.mean(all_preds, axis=0)


def save_xgboost(model, path):
    joblib.dump(model, path)
    print(f"XGBoost saved: {path}")


def load_xgboost(path):
    return joblib.load(path)


def save_ensemble_config(weights, path):
    config = {'weights': weights.tolist()}
    with open(path, 'w') as f:
        json.dump(config, f)
    print(f"Ensemble config saved: {path}")


def load_ensemble_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return np.array(config['weights'])


def main():
    from train_panderm import PanDermDataset, get_panderm_transforms
    from torch.utils.data import DataLoader
    from models_panderm import create_panderm_model
    from utils import get_device, load_checkpoint
    
    device = get_device()
    
    print("Loading data...")
    train_df = pd.read_csv(PREPROCESSED_DIR / 'train_data.csv')
    val_df = pd.read_csv(PREPROCESSED_DIR / 'val_data.csv')
    
    val_transform = get_panderm_transforms(IMAGE_CONFIG_PANDERM['image_size'], augment=False)
    train_dataset = PanDermDataset(train_df, transform=val_transform)
    val_dataset = PanDermDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Loading PanDerm model...")
    panderm = create_panderm_model(pretrained=False)
    checkpoint_path = Path(MODELS_DIR) / 'panderm_best.pth'
    if checkpoint_path.exists():
        load_checkpoint(panderm, filepath=checkpoint_path, device=device)
    panderm = panderm.to(device).eval()
    
    print("Extracting training features...")
    train_features, train_labels = extract_features(panderm, train_loader, device)
    print(f"Train features: {train_features.shape}")
    
    print("Extracting validation features...")
    val_features, val_labels = extract_features(panderm, val_loader, device)
    print(f"Val features: {val_features.shape}")
    
    xgb_model = train_xgboost(train_features, train_labels, val_features, val_labels)
    save_xgboost(xgb_model, Path(MODELS_DIR) / 'xgboost_stacking.pkl')
    
    print("\nOptimizing ensemble weights...")
    train_panderm_probs = train_features[:, -11:]
    val_panderm_probs = val_features[:, -11:]
    
    val_xgb_probs = predict_xgboost(xgb_model, val_features)
    
    weights = optimize_ensemble_weights(val_panderm_probs, None, val_xgb_probs, val_labels)
    save_ensemble_config(weights, Path(MODELS_DIR) / 'ensemble_weights.json')
    
    print("\nXGBoost stacking completed!")


if __name__ == "__main__":
    main()

