"""
Script to prepare test data and generate submission
"""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *


def prepare_test_data():
    """
    Prepare test data CSV with metadata for inference
    
    This function creates a test data CSV similar to the training data format,
    but for test samples. It includes lesion_id, image paths, and metadata features.
    """
    print("Preparing test data...")
    
    # Load test metadata (assuming similar format to training metadata)
    test_metadata_path = DATASET_DIR / 'MILK10k_Test_Metadata.csv'
    
    if not test_metadata_path.exists():
        print(f"Warning: {test_metadata_path} not found.")
        print("Creating test data from directory structure...")
        
        # Get all test image directories
        test_dirs = sorted([d for d in TEST_INPUT_DIR.iterdir() if d.is_dir()])
        
        # Create basic test DataFrame with lesion_id
        test_data = []
        for test_dir in test_dirs:
            lesion_id = test_dir.name
            if not lesion_id.startswith('IL_'):
                continue  # Skip non-lesion directories
            
            # Find images in lesion directory
            image_files = sorted(list(test_dir.glob('*.jpg')))
            if len(image_files) < 2:
                print(f"Warning: Lesion {lesion_id} has {len(image_files)} images, expected 2")
                continue
            
            test_data.append({
                'lesion_id': lesion_id,
                'clinical_image_path': str(image_files[0]),
                'dermoscopic_image_path': str(image_files[1]),
                'age_approx': 50.0,
                'sex': 'unknown',
                'site': 'unknown',
                'skin_tone_class': 3.0
            })
        
        test_df = pd.DataFrame(test_data)
        print(f"Created test data with {len(test_df):,} samples (without metadata)")
    
    else:
        # Load test metadata
        test_metadata = pd.read_csv(test_metadata_path)
        print(f"Loaded test metadata: {len(test_metadata):,} rows (958 images from 479 lesions)")
        
        # Group by lesion_id and process each lesion
        lesion_data = []
        for lesion_id, group in test_metadata.groupby('lesion_id'):
            # Separate clinical and dermoscopic images
            clinical_row = group[group['image_type'] == 'clinical: close-up']
            dermoscopic_row = group[group['image_type'] == 'dermoscopic']
            
            if len(clinical_row) == 0 or len(dermoscopic_row) == 0:
                print(f"Warning: Lesion {lesion_id} missing clinical or dermoscopic image")
                continue
            
            clinical_row = clinical_row.iloc[0]
            dermoscopic_row = dermoscopic_row.iloc[0]
            
            # Build image paths
            clinical_path = str(TEST_INPUT_DIR / lesion_id / f"{clinical_row['isic_id']}.jpg")
            dermoscopic_path = str(TEST_INPUT_DIR / lesion_id / f"{dermoscopic_row['isic_id']}.jpg")
            
            # Create lesion entry with properly prefixed MONET features
            lesion_entry = {
                'lesion_id': lesion_id,
                'clinical_image_path': clinical_path,
                'dermoscopic_image_path': dermoscopic_path,
                'clinical_isic_id': clinical_row['isic_id'],
                'dermoscopic_isic_id': dermoscopic_row['isic_id'],
                'age_approx': clinical_row.get('age_approx', 50.0),
                'sex': clinical_row.get('sex', 'unknown'),
                'site': clinical_row.get('site', 'unknown'),
                'skin_tone_class': clinical_row.get('skin_tone_class', 3.0),
            }
            
            # Add MONET features with proper prefixes (to match training data format)
            monet_cols = [col for col in clinical_row.index if col.startswith('MONET_')]
            for col in monet_cols:
                lesion_entry[f'clinical_{col}'] = clinical_row[col]
                lesion_entry[f'dermoscopic_{col}'] = dermoscopic_row[col]
            
            lesion_data.append(lesion_entry)
        
        test_df = pd.DataFrame(lesion_data)
        print(f"Unique lesions: {len(test_df):,}")
        
        # Fill NaN values
        test_df = test_df.copy()
        test_df['age_approx'] = test_df['age_approx'].fillna(50.0)
        test_df['sex'] = test_df['sex'].fillna('unknown')
        test_df['site'] = test_df['site'].fillna('unknown')
        test_df['skin_tone_class'] = test_df['skin_tone_class'].fillna(3.0)
    
    # Save processed test data
    output_path = PREPROCESSED_DIR / 'test_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    
    print(f"Test data saved: {output_path}")
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    print("\nTest data preview:")
    print(test_df.head())
    
    return test_df


def generate_submission(model_path=None, use_tta=False):
    """
    Generate submission file using trained model
    
    Args:
        model_path: Path to trained model checkpoint (default: best_model.pth)
        use_tta: Whether to use Test Time Augmentation
    """
    from inference import Predictor, create_submission
    from utils import get_device
    
    # Set default model path
    if model_path is None:
        model_path = MODELS_DIR / 'best_model.pth'
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first or specify a valid model path.")
        return
    
    # Prepare test data
    test_df = prepare_test_data()
    
    # Get device
    device = get_device()
    
    # Create predictor
    print("\nLoading model for inference...")
    predictor = Predictor(
        model_path=model_path,
        device=device
    )
    
    # Generate predictions
    if use_tta:
        print("\nGenerating predictions with TTA...")
        predictions = predictor.predict_with_tta(
            test_df=test_df,
            batch_size=32,
            num_workers=4,
            n_tta=5
        )
        output_filename = 'submission_tta.csv'
    else:
        print("\nGenerating predictions...")
        predictions = predictor.predict(
            test_df=test_df,
            batch_size=32,
            num_workers=4
        )
        output_filename = 'submission.csv'
    
    # Create submission
    output_path = RESULTS_DIR / output_filename
    submission = create_submission(
        predictions=predictions,
        test_df=test_df,
        output_path=output_path,
        threshold=0.5
    )
    
    print(f"\n✅ Submission generated successfully!")
    print(f"Submission file: {output_path}")
    
    return submission


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate submission for MILK10k')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (default: models/best_model.pth)')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation')
    parser.add_argument('--prepare_only', action='store_true',
                        help='Only prepare test data without generating submission')
    
    args = parser.parse_args()
    
    if args.prepare_only:
        # Only prepare test data
        prepare_test_data()
    else:
        # Generate submission
        generate_submission(
            model_path=args.model_path,
            use_tta=args.use_tta
        )


if __name__ == "__main__":
    main()
