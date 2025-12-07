# 🩺 MILK10k Skin Lesion Classification

Deep Learning project for multi-label skin lesion classification using the MILK10k dataset with 11 diagnostic categories.

## 📋 Objective

Build a multi-label classification model for 11 skin lesion types:

| Code | Description |
|------|-------------|
| AKIEC | Actinic keratosis / intraepidermal carcinoma |
| BCC | Basal cell carcinoma |
| BEN_OTH | Other benign proliferations |
| BKL | Benign keratinocytic lesion |
| DF | Dermatofibroma |
| INF | Inflammatory and infectious conditions |
| MAL_OTH | Other malignant proliferations |
| MEL | Melanoma |
| NV | Melanocytic nevus |
| SCCKA | Squamous cell carcinoma / keratoacanthoma |
| VASC | Vascular lesions and hemorrhage |

## 📊 Dataset

| Split | Lesions | Images |
|-------|---------|--------|
| Training | 5,240 | 10,480 |
| Test | 479 | 958 |

- Each lesion has 2 images: **Clinical close-up** + **Dermoscopic**
- Metadata: Age, sex, skin tone, anatomical site, MONET scores

## 🏗️ Project Structure

```
DEEP_LEARNING/
├── dataset/                          # Raw data
│   ├── MILK10k_Training_GroundTruth.csv
│   ├── MILK10k_Training_Metadata.csv
│   ├── MILK10k_Training_Supplement.csv
│   ├── MILK10k_Test_Metadata.csv
│   ├── MILK10k_Training_Input/       # Training images
│   └── MILK10k_Test_Input/           # Test images
│
├── preprocessed_data/                # Processed data
│   ├── train_data.csv               # 4,192 samples
│   ├── val_data.csv                 # 1,048 samples
│   ├── test_data.csv                # 479 samples (generated)
│   └── class_weights.json
│
├── src/                              # Source code
│   ├── config.py                     # Configuration
│   ├── data_preprocessing.py         # Data preprocessing
│   ├── dataset.py                    # Dataset & DataLoader
│   ├── models.py                     # Model architectures
│   ├── train.py                      # Training pipeline
│   ├── inference.py                  # Inference pipeline
│   ├── generate_submission.py        # Submission generator
│   ├── evaluate.py                   # Evaluation metrics
│   └── utils.py                      # Utilities
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   └── Train_MILK10k_Colab.ipynb     # Google Colab training
│
├── models/                           # Saved models
│   ├── best_model.pth               # Best model checkpoint
│   └── training_history.csv         # Training history
│
├── results/                          # Results
│   └── submission.csv               # Generated submission
│
├── logs/                             # TensorBoard logs
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n milk10k python=3.10
conda activate milk10k

# Install PyTorch with CUDA (check your CUDA version with nvidia-smi)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 2. Data Preprocessing

```bash
python src/data_preprocessing.py
```

Output:
- `preprocessed_data/train_data.csv`: 4,192 samples
- `preprocessed_data/val_data.csv`: 1,048 samples  
- `preprocessed_data/class_weights.json`: Class weights for imbalance

### 3. Training

**Option A: Local GPU Training**
```bash
python src/train.py
```

**Option B: Google Colab** (recommended if no local GPU)
- Upload `notebooks/Train_MILK10k_Colab.ipynb` to Colab
- Set runtime to GPU (T4 or A100)
- Follow the notebook instructions

Monitor training:
```bash
tensorboard --logdir=logs
```

### 4. Generate Submission

```bash
# Standard inference
python src/generate_submission.py --model_path models/best_model.pth

# With Test Time Augmentation (better accuracy, slower)
python src/generate_submission.py --model_path models/best_model.pth --use_tta
```

Output: `results/submission.csv`

## 🔧 Model Architecture

| Component | Configuration |
|-----------|--------------|
| Backbone | EfficientNet-B3 (ImageNet pretrained) |
| Input Size | 384×384 RGB |
| Fusion Strategy | Early fusion (concatenate clinical + dermoscopic → 6 channels) |
| Metadata | 18 features (4 clinical + 14 MONET scores) |
| Output | 11-class multi-label (sigmoid) |
| Parameters | ~12M |

### Training Configuration

```python
MODEL_CONFIG = {
    'architecture': 'efficientnet_b3',
    'pretrained': True,
    'use_metadata': True,
    'metadata_dim': 18,
    'dropout': 0.3
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'early_stopping_patience': 15,
    'mixed_precision': True  # AMP for faster training
}

LOSS_CONFIG = {
    'use_focal_loss': True,
    'focal_gamma': 2.0,
    'use_class_weights': True
}
```

### Data Augmentation

**Training:**
- RandomRotate90, HorizontalFlip, VerticalFlip
- ShiftScaleRotate, ElasticTransform, GridDistortion
- ColorJitter, GaussNoise, GaussianBlur
- CoarseDropout

**Validation/Test:** Resize + Normalize only

## 📈 Evaluation

- **Primary Metric**: Macro F1 Score
- **Threshold**: 0.5 for binary prediction
- **Multi-label**: Each lesion can belong to multiple categories

## ⚙️ System Requirements

| Type | Minimum | Recommended |
|------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| GPU | - | RTX 3060+ (8GB+ VRAM) |
| Storage | 20GB | 50GB+ |

### Training Time Estimates

| GPU | Time (100 epochs) |
|-----|------------------|
| RTX 3060 (12GB) | ~10-15 hours |
| RTX 3090 (24GB) | ~5-8 hours |
| A100 (40GB) | ~3-5 hours |
| T4 (Colab) | ~15-20 hours |

## 🎯 Project Status

- [x] EDA & Data Understanding
- [x] Data Preprocessing Pipeline
- [x] Baseline Model (EfficientNet-B3)
- [x] Multi-input Architecture (Early Fusion + Metadata)
- [x] Training Pipeline (Focal Loss, AMP, Early Stopping)
- [x] Inference & Submission Generator
- [x] Google Colab Training Notebook
- [ ] Hyperparameter Tuning
- [ ] Ensemble Methods

## 📝 Notes

- **Class Imbalance**: Handled via Focal Loss + class weights
- **Multi-label**: Uses BCE Loss (not CrossEntropy)
- **Dual-image Input**: Early fusion strategy (6-channel input)
- **Metadata Integration**: Concatenated with image features before classifier
