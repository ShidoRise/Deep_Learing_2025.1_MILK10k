# 🚀 TRAINING GUIDE

## 📋 Prerequisites

### Yêu cầu phần cứng:
- **GPU**: NVIDIA RTX 3060+ (8GB VRAM trở lên) - **BẮT BUỘC**
- **RAM**: 16GB+
- **Storage**: 50GB+ dung lượng trống
- **CPU**: 8+ cores (khuyến nghị)

### Yêu cầu phần mềm:
- Python 3.10+
- CUDA 11.8 hoặc 12.1
- Conda hoặc virtualenv

---

## 🛠️ Setup Environment

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd DEEP_LEARNING
```

### Bước 2: Tạo Conda Environment

```bash
# Tạo environment mới
conda create -n milk10k python=3.10
conda activate milk10k
```

### Bước 3: Cài đặt PyTorch với CUDA

**Quan trọng**: Cài PyTorch với CUDA support trước khi cài các package khác!

```bash
# Kiểm tra CUDA version
nvidia-smi

# Cho CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cho CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Kiểm tra PyTorch đã nhận GPU chưa
python -c "import torch; print(torch.cuda.is_available())"
# Phải in ra: True
```

### Bước 4: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Chuẩn bị Dataset

### Option 1: Sử dụng preprocessed data có sẵn

Nếu có member đã chạy preprocessing, download các file sau:
- `preprocessed_data/train_data.csv`
- `preprocessed_data/val_data.csv`
- `preprocessed_data/class_weights.json`

### Option 2: Chạy preprocessing từ đầu

```bash
# Đảm bảo có dataset gốc trong thư mục dataset/
python src/data_preprocessing.py
```

Kết quả:
```
preprocessed_data/
├── train_data.csv          # 4,192 samples
├── val_data.csv            # 1,048 samples
└── class_weights.json      # Class weights cho Focal Loss
```

---

## 🏋️ Training

### Kiểm tra cấu hình trước khi train

Mở file `src/config.py` và xác nhận các settings:

```python
# Model config
MODEL_CONFIG = {
    'architecture': 'efficientnet_b3',
    'pretrained': True,
    'use_metadata': True,
    'metadata_dim': 18,
    'dropout': 0.3
}

# Training config
TRAIN_CONFIG = {
    'batch_size': 16,          # Giảm xuống 8 nếu OOM
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'early_stopping_patience': 15,
    'mixed_precision': True    # Quan trọng: giảm VRAM usage
}
```

### Start Training

```bash
python src/train.py
```

### Giám sát Training với TensorBoard

Mở terminal mới:

```bash
conda activate milk10k
tensorboard --logdir=logs

# Mở browser: http://localhost:6006
```

TensorBoard sẽ hiển thị:
- Training/Validation loss
- Macro F1 Score
- Micro F1 Score
- Per-class F1 scores (11 classes)
- Learning rate schedule

---

## 📊 Training Progress

### Thời gian dự kiến:

| GPU | Batch Size | Time per Epoch | Total Time (100 epochs) |
|-----|------------|----------------|-------------------------|
| RTX 3060 (12GB) | 16 | ~8-10 min | ~10-15 giờ |
| RTX 3070 (8GB) | 8-12 | ~10-12 min | ~12-18 giờ |
| RTX 3080 (10GB) | 16 | ~5-7 min | ~6-10 giờ |
| RTX 3090 (24GB) | 32 | ~3-5 min | ~5-8 giờ |
| RTX 4090 (24GB) | 32 | ~2-3 min | ~3-5 giờ |
| A100 (40GB) | 64 | ~1-2 min | ~2-3 giờ |

**Lưu ý**: Early stopping có thể dừng training sớm nếu F1 không cải thiện sau 15 epochs.

### Checkpoints được lưu:

```
models/
├── best_model.pth              # Model tốt nhất (theo Macro F1)
├── checkpoint_epoch_5.pth      # Checkpoint mỗi 5 epochs
├── checkpoint_epoch_10.pth
├── ...
└── training_history.csv        # Lịch sử training
```

---

## 🚨 Troubleshooting

### 1. CUDA Out of Memory (OOM)

**Triệu chứng**: `RuntimeError: CUDA out of memory`

**Giải pháp**:
```python
# Trong src/config.py, giảm batch_size
TRAIN_CONFIG = {
    'batch_size': 8,  # Giảm từ 16 xuống 8
    # ...
}
```

Hoặc:
```python
# Giảm image_size
IMAGE_CONFIG = {
    'image_size': 224,  # Giảm từ 384 xuống 224
    # ...
}
```

### 2. Training quá chậm

**Kiểm tra**:
- Mixed precision có đang bật không? `TRAIN_CONFIG['mixed_precision'] = True`
- `num_workers` có phù hợp? Thử `num_workers = 4` hoặc `8`
- GPU có đang chạy các tiến trình khác không? Kiểm tra `nvidia-smi`

### 3. Loss không giảm

**Kiểm tra**:
- Class weights có load đúng không?
- Learning rate có quá cao? Thử giảm xuống `5e-5`
- Dữ liệu có được chuẩn hóa đúng không?

### 4. Validation F1 thấp

**Thử**:
- Training thêm epochs
- Thay đổi augmentation
- Thử fusion strategy khác (`late` thay vì `early`)
- Tăng dropout để tránh overfitting

---

## 💾 Sau khi Training xong

### 1. Kiểm tra kết quả

```python
import pandas as pd

# Xem training history
history = pd.read_csv('models/training_history.csv')
print(history.tail(10))

# Xem best F1 score
print(f"Best Macro F1: {history['val_f1_macro'].max():.4f}")
```

### 2. Generate Submission

```bash
# Prediction cơ bản
python src/generate_submission.py --model_path models/best_model.pth

# Với Test Time Augmentation (khuyến nghị)
python src/generate_submission.py --model_path models/best_model.pth --use_tta
```

Kết quả:
- `results/submission.csv`
- `results/submission_tta.csv`

### 3. Share Model với Team

**Option 1: Git LFS** (nếu repo hỗ trợ)
```bash
git lfs install
git lfs track "*.pth"
git add models/best_model.pth
git commit -m "Add trained model (F1: 0.XXXX)"
git push
```

**Option 2: Google Drive / Dropbox**
```bash
# Upload models/best_model.pth lên Drive
# Share link trong team chat
```

**Option 3: Hugging Face Hub**
```bash
pip install huggingface_hub
python scripts/upload_to_hf.py  # (tạo script riêng nếu cần)
```

---

## 📈 Expected Results

### Target Metrics (theo literature):

- **Baseline F1 (EfficientNet-B3)**: 0.70 - 0.75
- **With metadata fusion**: 0.75 - 0.80
- **With TTA + ensemble**: 0.80 - 0.85+

### Training curve mẫu:

```
Epoch 1/100 - 10m 23s
  Train Loss: 0.3521
  Val Loss:   0.2987
  Val F1 (Macro): 0.6234
  Learning Rate: 0.000100

Epoch 10/100 - 9m 45s
  Train Loss: 0.1834
  Val Loss:   0.1923
  Val F1 (Macro): 0.7123
  Learning Rate: 0.000092

...

Epoch 45/100 - 9m 38s
  Train Loss: 0.0923
  Val Loss:   0.1456
  Val F1 (Macro): 0.7812
  ✅ Best model saved! F1: 0.7812

Early stopping triggered at epoch 60
```

---

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra lại các bước trong guide này
2. Search error message trên Google/Stack Overflow
3. Hỏi trong team chat
4. Tạo issue trên GitHub repo

---

## 📝 Notes cho người Training

### Quan trọng:
- [ ] Commit training history CSV
- [ ] Commit best model (hoặc upload Drive + share link)
- [ ] Ghi lại best F1 score trong README
- [ ] Screenshot TensorBoard curves (loss, F1)
- [ ] Note lại training time và GPU used

### Checkpoint:
```markdown
## Training Results

- **Date**: YYYY-MM-DD
- **GPU**: RTX 3090
- **Training Time**: 6.5 hours
- **Best Epoch**: 45
- **Best Macro F1**: 0.7812
- **Best Micro F1**: 0.8156
- **Model**: models/best_model.pth
- **Download**: [Google Drive Link]
```

---

Happy Training! 🚀
