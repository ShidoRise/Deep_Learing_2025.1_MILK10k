"""
Data preprocessing pipeline for MILK10k dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from utils import set_seed, create_directories


class MILK10kPreprocessor:
    def __init__(self):
        self.train_gt = None
        self.train_metadata = None
        self.train_supplement = None
        self.processed_df = None
        
    def load_data(self):
        print("Loading data files...")
        
        self.train_gt = pd.read_csv(TRAIN_GT_FILE)
        print(f"Ground truth: {self.train_gt.shape}")
        
        self.train_metadata = pd.read_csv(TRAIN_METADATA_FILE)
        print(f"Metadata: {self.train_metadata.shape}")
        
        self.train_supplement = pd.read_csv(TRAIN_SUPPLEMENT_FILE)
        print(f"Supplement: {self.train_supplement.shape}")
        
        return self
    
    def merge_data(self):
        print("\nMerging data sources...")
        
        clinical_df = self.train_metadata[
            self.train_metadata['image_type'] == 'clinical: close-up'
        ].copy()
        clinical_df = clinical_df.rename(columns={
            'isic_id': 'clinical_isic_id'
        })
        
        dermoscopic_df = self.train_metadata[
            self.train_metadata['image_type'] == 'dermoscopic'
        ].copy()
        dermoscopic_df = dermoscopic_df.rename(columns={
            'isic_id': 'dermoscopic_isic_id'
        })
        
        clinical_cols = ['lesion_id', 'clinical_isic_id'] + [
            col for col in clinical_df.columns if col.startswith('MONET_')
        ]
        clinical_cols = [col for col in clinical_cols if col in clinical_df.columns]
        
        dermoscopic_cols = ['lesion_id', 'dermoscopic_isic_id'] + [
            col for col in dermoscopic_df.columns if col.startswith('MONET_')
        ]
        dermoscopic_cols = [col for col in dermoscopic_cols if col in dermoscopic_df.columns]
        
        clinical_monet = clinical_df[clinical_cols].copy()
        for col in clinical_monet.columns:
            if col.startswith('MONET_'):
                clinical_monet = clinical_monet.rename(columns={
                    col: f'clinical_{col}'
                })
        
        dermoscopic_monet = dermoscopic_df[dermoscopic_cols].copy()
        for col in dermoscopic_monet.columns:
            if col.startswith('MONET_'):
                dermoscopic_monet = dermoscopic_monet.rename(columns={
                    col: f'dermoscopic_{col}'
                })
        
        metadata_cols = ['lesion_id', 'age_approx', 'sex', 'skin_tone_class', 'site']
        clinical_metadata = clinical_df[metadata_cols].copy()
        
        self.processed_df = self.train_gt.copy()
        self.processed_df = self.processed_df.merge(clinical_metadata, on='lesion_id', how='left')
        self.processed_df = self.processed_df.merge(clinical_monet, on='lesion_id', how='left')
        self.processed_df = self.processed_df.merge(dermoscopic_monet, on='lesion_id', how='left')
        
        print(f"Merged data shape: {self.processed_df.shape}")
        
        return self
    
    def add_image_paths(self):
        print("\nAdding image paths...")
        
        clinical_paths = []
        dermoscopic_paths = []
        
        for _, row in tqdm(self.processed_df.iterrows(), total=len(self.processed_df)):
            lesion_id = row['lesion_id']
            lesion_dir = TRAIN_INPUT_DIR / lesion_id
            
            if lesion_dir.exists():
                clinical_img = None
                dermoscopic_img = None
                
                for img_file in lesion_dir.glob('*.jpg'):
                    isic_id = img_file.stem
                    
                    if row['clinical_isic_id'] == isic_id:
                        clinical_img = str(img_file)
                    elif row['dermoscopic_isic_id'] == isic_id:
                        dermoscopic_img = str(img_file)
                
                clinical_paths.append(clinical_img)
                dermoscopic_paths.append(dermoscopic_img)
            else:
                clinical_paths.append(None)
                dermoscopic_paths.append(None)
        
        self.processed_df['clinical_image_path'] = clinical_paths
        self.processed_df['dermoscopic_image_path'] = dermoscopic_paths
        
        missing = (
            self.processed_df['clinical_image_path'].isna() | 
            self.processed_df['dermoscopic_image_path'].isna()
        ).sum()
        
        if missing > 0:
            print(f"Warning: {missing} lesions have missing images")
            self.processed_df = self.processed_df.dropna(
                subset=['clinical_image_path', 'dermoscopic_image_path']
            )
        
        print(f"Final dataset size: {len(self.processed_df)} lesions")
        
        return self
    
    def analyze_class_distribution(self):
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        class_counts = self.processed_df[DIAGNOSIS_CATEGORIES].sum().sort_values(ascending=False)
        
        print("\nNumber of cases per diagnosis:")
        for category, count in class_counts.items():
            percentage = (count / len(self.processed_df)) * 100
            print(f"  {category:10s}: {count:5d} ({percentage:5.2f}%)")
        
        multi_label_count = (self.processed_df[DIAGNOSIS_CATEGORIES].sum(axis=1) > 1).sum()
        print(f"\nMulti-label cases: {multi_label_count} ({multi_label_count/len(self.processed_df)*100:.2f}%)")
        
        return class_counts
    
    def split_data(self, train_ratio=0.8, random_seed=42):
        print(f"\nSplitting data (train: {train_ratio:.0%}, val: {1-train_ratio:.0%})...")
        
        set_seed(random_seed)
        
        primary_diagnosis = self.processed_df[DIAGNOSIS_CATEGORIES].idxmax(axis=1)
        
        train_df, val_df = train_test_split(
            self.processed_df,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=primary_diagnosis
        )
        
        print(f"Training set: {len(train_df)} lesions")
        print(f"Validation set: {len(val_df)} lesions")
        
        return train_df, val_df
    
    def save_processed_data(self, train_df, val_df):
        print("\nSaving processed data...")
        
        create_directories([PREPROCESSED_DIR])
        
        train_path = PREPROCESSED_DIR / "train_data.csv"
        val_path = PREPROCESSED_DIR / "val_data.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"Training data saved: {train_path}")
        print(f"Validation data saved: {val_path}")
        
        full_path = PREPROCESSED_DIR / "full_processed_data.csv"
        self.processed_df.to_csv(full_path, index=False)
        print(f"Full processed data saved: {full_path}")
        
        return self
    
    def get_class_weights(self, train_df):
        print("\nCalculating class weights...")
        
        class_counts = train_df[DIAGNOSIS_CATEGORIES].sum()
        total_samples = len(train_df)
        
        class_weights = {}
        for category in DIAGNOSIS_CATEGORIES:
            weight = total_samples / (len(DIAGNOSIS_CATEGORIES) * class_counts[category])
            class_weights[category] = weight
            print(f"  {category:10s}: {weight:.4f}")
        
        import json
        weights_path = PREPROCESSED_DIR / "class_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f, indent=4)
        
        print(f"Class weights saved: {weights_path}")
        
        return class_weights


def main():
    print("="*60)
    print("MILK10k DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    preprocessor = MILK10kPreprocessor()
    
    preprocessor.load_data()
    preprocessor.merge_data()
    preprocessor.add_image_paths()
    
    class_counts = preprocessor.analyze_class_distribution()
    
    train_df, val_df = preprocessor.split_data(
        train_ratio=DATA_SPLIT['train_ratio'],
        random_seed=DATA_SPLIT['random_seed']
    )
    
    class_weights = preprocessor.get_class_weights(train_df)
    
    preprocessor.save_processed_data(train_df, val_df)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED!")
    print("="*60)
    print(f"\nProcessed data saved to: {PREPROCESSED_DIR}")
    print("\nNext steps:")
    print("1. Run EDA notebook: notebooks/01_EDA.ipynb")
    print("2. Train model: python src/train.py")


if __name__ == "__main__":
    main()
