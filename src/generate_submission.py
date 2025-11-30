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
    but for test samples. It includes isic_id and metadata features.
    """
    print("Preparing test data...")
    
    # Load test metadata (assuming similar format to training metadata)
    test_metadata_path = DATASET_DIR / 'MILK10k_Test_Metadata.csv'
    
    if not test_metadata_path.exists():
        print(f"Warning: {test_metadata_path} not found.")
        print("Creating test data from directory structure...")
        
        # Get all test image directories
        test_dirs = sorted([d for d in TEST_INPUT_DIR.iterdir() if d.is_dir()])
        
        # Create basic test DataFrame with isic_id
        test_data = []
        for test_dir in test_dirs:
            isic_id = test_dir.name
            test_data.append({'isic_id': isic_id})
        
        test_df = pd.DataFrame(test_data)
        
        # Add default metadata values (will be normalized in dataset)
        test_df['age_approx'] = 50.0  # Default age
        test_df['sex'] = 'unknown'  # Default sex
        test_df['anatom_site_general'] = 'unknown'  # Default site
        test_df['tbp_lv_location_simple'] = 'unknown'  # Default location
        
        print(f"Created test data with {len(test_df):,} samples")
    
    else:
        # Load test metadata
        test_metadata = pd.read_csv(test_metadata_path)
        print(f"Loaded test metadata: {len(test_metadata):,} samples")
        
        # Ensure required columns exist
        required_cols = ['isic_id', 'age_approx', 'sex', 'anatom_site_general']
        
        for col in required_cols:
            if col not in test_metadata.columns:
                if col == 'age_approx':
                    test_metadata[col] = 50.0
                else:
                    test_metadata[col] = 'unknown'
        
        test_df = test_metadata
    
    # Save processed test data
    output_path = PREPROCESSED_DIR / 'test_data.csv'
    test_df.to_csv(output_path, index=False)
    
    print(f"Test data saved: {output_path}")
    print(f"Shape: {test_df.shape}")
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
