# 🩺 MILK10k Skin Lesion Classification

Dự án Deep Learning phân loại tổn thương da sử dụng dataset MILK10k với 11 loại chẩn đoán.

## 📋 Mục tiêu

Xây dựng mô hình phân loại đa nhãn (multi-label classification) cho 11 loại tổn thương da:
- AKIEC: Actinic keratosis / intraepidermal carcinoma
- BCC: Basal cell carcinoma
- BEN_OTH: Other benign proliferations
- BKL: Benign keratinocytic lesion
- DF: Dermatofibroma
- INF: Inflammatory and infectious conditions
- MAL_OTH: Other malignant proliferations
- MEL: Melanoma
- NV: Melanocytic nevus
- SCCKA: Squamous cell carcinoma / keratoacanthoma
- VASC: Vascular lesions and hemorrhage

## 📊 Dataset

- **Training**: 5,240 lesions (10,480 images)
- **Test**: 479 lesions (958 images)
- Mỗi lesion có 2 ảnh: Clinical close-up + Dermatoscopic
- Metadata: Age, sex, skin tone, anatomical site, MONET scores

## 🏗️ Cấu trúc Project

```
DEEP_LEARNING/
├── dataset/                          # Dữ liệu gốc
│   ├── MILK10k_Training_GroundTruth.csv
│   ├── MILK10k_Training_Metadata.csv
│   ├── MILK10k_Training_Supplement.csv
│   ├── MILK10k_Training_Input/       # Ảnh training
│   └── MILK10k_Test_Input/           # Ảnh test
│
├── preprocessed_data/                # Dữ liệu đã xử lý
│   ├── train/
│   ├── val/
│   └── metadata.csv
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration
│   ├── data_preprocessing.py         # Tiền xử lý dữ liệu
│   ├── dataset.py                    # Dataset & DataLoader
│   ├── augmentation.py               # Data augmentation
│   ├── models.py                     # Kiến trúc mô hình
│   ├── train.py                      # Training pipeline
│   ├── evaluate.py                   # Evaluation metrics
│   └── utils.py                      # Utilities
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb        # Data preprocessing
│   ├── 03_Training.ipynb             # Model training
│   └── 04_Evaluation.ipynb           # Model evaluation
│
├── models/                           # Saved models
│   └── checkpoints/
│
├── results/                          # Kết quả
│   ├── predictions/
│   ├── visualizations/
│   └── metrics/
│
├── logs/                             # Training logs
│
├── requirements.txt
└── README.md
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
# Tạo conda environment
conda create -n milk10k python=3.10
conda activate milk10k

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Tiền xử lý dữ liệu

```bash
python src/data_preprocessing.py
```

Kết quả:
- `preprocessed_data/train_data.csv`: 4,192 samples
- `preprocessed_data/val_data.csv`: 1,048 samples  
- `preprocessed_data/class_weights.json`: Trọng số cho class imbalance

### 3. Khám phá dữ liệu (Optional)

```bash
jupyter notebook notebooks/EDA.ipynb
```

### 4. Training model

**Lưu ý**: Training yêu cầu GPU mạnh (recommended: RTX 3060+, 8GB+ VRAM)

```bash
python src/train.py
```

Cấu hình mặc định:
- Model: EfficientNet-B3
- Image size: 384×384
- Batch size: 16
- Epochs: 100 (với early stopping patience=15)
- Loss: Focal Loss với class weights
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Mixed precision training (AMP)
- TensorBoard logging

Kết quả training:
- `models/best_model.pth`: Model tốt nhất theo Macro F1
- `models/training_history.csv`: Lịch sử training
- `logs/`: TensorBoard logs

Xem training progress:
```bash
tensorboard --logdir=logs
```

### 5. Inference & Generate Submission

Sau khi training xong, tạo file submission:

```bash
# Prediction thông thường
python src/generate_submission.py --model_path models/best_model.pth

# Với Test Time Augmentation (TTA) - tốt hơn nhưng chậm hơn
python src/generate_submission.py --model_path models/best_model.pth --use_tta
```

Kết quả:
- `results/submission.csv`: File submission chuẩn
- `results/submission_tta.csv`: File submission với TTA

### 6. Evaluation (Optional)

Đánh giá model trên validation set:

```bash
python src/evaluate.py --model_path models/best_model.pth
```

## 📈 Evaluation Metric

- **Primary**: Macro F1 Score
- **Threshold**: 0.5 cho binary prediction
- Multi-label: Một lesion có thể được dự đoán thuộc nhiều category

## 🧪 Chiến lược Preprocessing

1. **Image Processing**:
   - Resize về kích thước chuẩn (224x224 hoặc 384x384)
   - Normalization theo ImageNet stats
   - Color augmentation

2. **Data Fusion**:
   - Early fusion: Concatenate 2 ảnh
   - Late fusion: Ensemble predictions
   - Feature-level fusion

3. **Metadata Integration**:
   - MONET scores (ulceration, hair, vasculature, etc.)
   - Age, sex, skin tone, anatomical site
   - Concatenate với image features

4. **Data Augmentation**:
   - Random rotation, flip, crop
   - Color jittering
   - Cutout / MixUp

5. **Class Imbalance**:
   - Weighted loss function
   - Oversampling minority classes
   - Focal Loss

## 🎯 Roadmap

- [x] Phase 1: EDA & Data Understanding
- [x] Phase 2: Data Preprocessing Pipeline
- [x] Phase 3: Baseline Model (EfficientNet-B3)
- [x] Phase 4: Multi-input Architecture (Early Fusion + Metadata)
- [x] Phase 5: Training Pipeline với Focal Loss, AMP, Early Stopping
- [x] Phase 6: Inference & Submission Generator
- [ ] Phase 7: Hyperparameter Tuning
- [ ] Phase 8: Ensemble Methods
- [ ] Phase 9: Submit to MILK10k Benchmark

## 🔧 Technical Details

### Model Architecture
- **Backbone**: EfficientNet-B3 (pretrained on ImageNet)
- **Input**: 384×384 RGB images
- **Fusion Strategy**: Early fusion (concatenate clinical + dermoscopic images)
- **Metadata Integration**: Concatenate với image features trước classifier
- **Output**: 11-class multi-label classification
- **Total Parameters**: ~11.5M

### Training Configuration
```python
MODEL_CONFIG = {
    'architecture': 'efficientnet_b3',
    'pretrained': True,
    'use_metadata': True,
    'metadata_dim': 18,  # 4 clinical + 14 MONET scores
    'dropout': 0.3
}

TRAIN_CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'early_stopping_patience': 15,
    'mixed_precision': True
}

LOSS_CONFIG = {
    'use_focal_loss': True,
    'focal_gamma': 2.0,
    'use_class_weights': True
}
```

### Data Augmentation
- Training: RandomRotate90, HorizontalFlip, VerticalFlip, ShiftScaleRotate, ColorJitter, GaussNoise, CoarseDropout
- Validation/Test: Only Resize + Normalize

## 🤝 Hướng dẫn cho Team Members

### Nếu bạn có GPU mạnh để train:

1. **Clone repository**:
```bash
git clone <repository-url>
cd DEEP_LEARNING
```

2. **Setup environment**:
```bash
conda create -n milk10k python=3.10
conda activate milk10k
pip install -r requirements.txt
```

3. **Download dataset** và đặt vào thư mục `dataset/`

4. **Tiền xử lý dữ liệu** (nếu chưa có preprocessed_data):
```bash
python src/data_preprocessing.py
```

5. **Start training**:
```bash
python src/train.py
```

6. **Monitor training** với TensorBoard:
```bash
tensorboard --logdir=logs
```

7. **Generate submission** sau khi training xong:
```bash
python src/generate_submission.py --model_path models/best_model.pth --use_tta
```

8. **Push model về repository**:
```bash
# Lưu ý: model files rất lớn, cân nhắc dùng Git LFS hoặc upload lên Google Drive
git add models/best_model.pth
git commit -m "Add trained model checkpoint"
git push
```

### Nếu chỉ muốn test inference:

1. Download pretrained model từ team member
2. Chạy inference:
```bash
python src/generate_submission.py --model_path path/to/model.pth
```

## ⚙️ System Requirements

### Minimum (cho inference):
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (CPU inference chậm nhưng vẫn chạy được)

### Recommended (cho training):
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX 3060 hoặc cao hơn (8GB+ VRAM)
- Storage: 50GB+ free space

### Training Time Estimate:
- **RTX 3060 (12GB)**: ~10-15 giờ cho 100 epochs
- **RTX 3090 (24GB)**: ~5-8 giờ cho 100 epochs
- **A100 (40GB)**: ~3-5 giờ cho 100 epochs
- **MX570 (2GB)**: ~30-40 giờ (không khuyến khích)

## 📝 Notes

- Dataset cân bằng: Check phân bố các classes
- Multi-label: Sử dụng BCE Loss thay vì CrossEntropy
- Fusion strategy quan trọng cho dual-image input
