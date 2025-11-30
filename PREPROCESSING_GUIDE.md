# 📋 HƯỚNG DẪN TIỀN XỬ LÝ DỮ LIỆU - MILK10k PROJECT

## 🎯 Mục tiêu Preprocessing

Chuẩn bị dữ liệu MILK10k để training model Deep Learning cho bài toán phân loại đa nhãn (multi-label classification) 11 loại tổn thương da.

---

## 📊 Tổng quan Dataset

### Dữ liệu có sẵn:
1. **10,480 ảnh** từ 5,240 lesions (mỗi lesion có 2 ảnh)
   - Clinical close-up image
   - Dermoscopic image

2. **11 nhãn chẩn đoán** (binary multi-label):
   - AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC

3. **Metadata phong phú**:
   - Clinical: age, sex, skin_tone, anatomical_site
   - MONET scores: ulceration, hair, vasculature, erythema, pigmentation, gel, skin markings

---

## 🔄 Pipeline Preprocessing (Đã triển khai)

### **Bước 1: Chạy script preprocessing**

```bash
cd d:\PYTHON\DEEP_LEARNING
python src/data_preprocessing.py
```

**Script này sẽ thực hiện:**

✅ Load và merge 3 file CSV (GroundTruth, Metadata, Supplement)  
✅ Tạo mapping giữa lesion_id và đường dẫn ảnh  
✅ Xử lý MONET scores cho cả 2 loại ảnh  
✅ Phân tích phân bố classes (class distribution)  
✅ Split train/val theo tỷ lệ 80/20 với stratification  
✅ Tính class weights để xử lý imbalanced data  
✅ Lưu dữ liệu đã xử lý vào `preprocessed_data/`

**Output files:**
- `preprocessed_data/train_data.csv` - Training set
- `preprocessed_data/val_data.csv` - Validation set
- `preprocessed_data/full_processed_data.csv` - Full dataset
- `preprocessed_data/class_weights.json` - Class weights

---

## 🖼️ Chiến lược xử lý ảnh

### **1. Image Fusion Strategies**

Có 3 cách kết hợp 2 ảnh (clinical + dermoscopic):

#### **A. Early Fusion** (Đang sử dụng)
```
Clinical RGB (3 channels) + Dermoscopic RGB (3 channels) 
→ Concatenate → 6 channels input
→ Single CNN backbone
```

**Ưu điểm:**
- Đơn giản, nhanh
- Model học được tương quan giữa 2 ảnh ngay từ đầu

**Nhược điểm:**
- Cần modify first conv layer (in_channels=6)
- Khó tận dụng pretrained weights của conv1

#### **B. Late Fusion**
```
Clinical RGB → CNN1 → Features1 ─┐
                                  ├→ Concatenate → Classifier
Dermoscopic RGB → CNN2 → Features2 ─┘
```

**Ưu điểm:**
- Tận dụng được pretrained weights hoàn toàn
- Học đặc trưng riêng cho từng loại ảnh

**Nhược điểm:**
- Nhiều parameters hơn (2 backbones)
- Training chậm hơn

#### **C. Feature-level Fusion với Attention** (Advanced)
```
Clinical → CNN1 → Features1 ─┐
                              ├→ Attention Fusion → Classifier
Dermoscopic → CNN2 → Features2 ─┘
```

**Ưu điểm:**
- Model tự học trọng số cho từng loại ảnh
- Linh hoạt, hiệu quả cao

---

### **2. Image Preprocessing Pipeline**

#### **Training transforms** (với augmentation):
```python
- Resize to 384x384 (hoặc 224, 512)
- Random rotation ±20°
- Random horizontal/vertical flip
- ShiftScaleRotate
- Color jittering (brightness, contrast, saturation, hue)
- Gaussian noise/blur
- CoarseDropout (cutout)
- Normalize theo ImageNet stats
```

#### **Validation transforms** (không augmentation):
```python
- Resize to 384x384
- Normalize theo ImageNet stats
```

---

## 📈 Xử lý Class Imbalance

### **Vấn đề:**
Dataset có thể không cân bằng giữa các classes (một số bệnh hiếm hơn)

### **Giải pháp đã triển khai:**

1. **Class Weights**
   - Tính toán: `weight = total_samples / (n_classes × class_count)`
   - Lưu trong `class_weights.json`
   - Sử dụng trong loss function

2. **Stratified Split**
   - Train/val split theo primary diagnosis
   - Đảm bảo tỷ lệ các classes giống nhau

3. **Focal Loss** (có thể bật trong config)
   - Tập trung vào hard examples
   - Giảm weight cho easy examples

---

## 🔢 Metadata Integration

### **Metadata features được sử dụng:**

1. **Categorical features:**
   - `sex`: male=0, female=1
   - `site`: head_neck_face=0, upper_extremity=1, lower_extremity=2, trunk=3, ...

2. **Numerical features (normalized):**
   - `age_approx`: chia cho 100
   - `skin_tone_class`: chia cho 5 (range 0-5)

3. **MONET scores** (đã normalized 0-1):
   - clinical_MONET_ulceration_crust
   - clinical_MONET_hair
   - clinical_MONET_vasculature_vessels
   - clinical_MONET_erythema
   - clinical_MONET_pigmented
   - clinical_MONET_gel_water_drop_fluid_dermoscopy_liquid
   - clinical_MONET_skin_markings_pen_ink_purple_pen
   - (7 features tương tự cho dermoscopic)

**Tổng: ~18 features metadata**

### **Cách kết hợp metadata:**
```
Image features (từ CNN) → [batch, 1536]
Metadata features → FC layers → [batch, 64]
Concatenate → [batch, 1600] → Classifier
```

---

## 🎨 Data Augmentation (Chi tiết)

### **Geometric augmentations:**
- RandomRotate90
- HorizontalFlip / VerticalFlip
- ShiftScaleRotate (shift ±10%, scale ±15%, rotate ±20°)

### **Color augmentations:**
- ColorJitter (brightness, contrast, saturation, hue)
- Useful vì skin tone varies

### **Noise & blur:**
- GaussianNoise
- GaussianBlur / MotionBlur
- Simulate real-world image quality variations

### **Cutout:**
- CoarseDropout: Remove random patches
- Force model to use multiple regions
- Prevent overfitting to specific areas

### **Advanced (optional):**
- MixUp: Mix 2 images with labels
- CutMix: Cut and paste image regions
- Grid/Elastic distortion

---

## 📁 Cấu trúc dữ liệu sau preprocessing

```
preprocessed_data/
├── train_data.csv              # 4,192 lesions (80%)
├── val_data.csv                # 1,048 lesions (20%)
├── full_processed_data.csv     # 5,240 lesions (full)
└── class_weights.json          # Class weights cho training

Columns in CSV:
- lesion_id
- AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC (labels)
- clinical_isic_id, dermoscopic_isic_id
- clinical_image_path, dermoscopic_image_path
- age_approx, sex, skin_tone_class, site
- clinical_MONET_* (7 features)
- dermoscopic_MONET_* (7 features)
```

---

## ✅ Checklist sau khi chạy preprocessing

- [ ] Kiểm tra không có missing images
- [ ] Xác nhận train/val split ratio đúng
- [ ] Kiểm tra class distribution trong train và val
- [ ] Review class weights (không nên quá lệch)
- [ ] Kiểm tra metadata không có NaN values
- [ ] Visualize một số samples với labels

---

## 🚀 Next Steps

### **1. Exploratory Data Analysis (EDA)**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

Phân tích:
- Class distribution visualization
- Image quality check
- Metadata correlation analysis
- Identify potential issues

### **2. Test DataLoader**
```bash
python src/dataset.py
```

Kiểm tra:
- Image loading works
- Transforms applied correctly
- Batch dimensions correct
- Metadata integration works

### **3. Train Baseline Model**
```bash
python src/train.py
```

---

## 🔧 Tuning Parameters

Có thể điều chỉnh trong `src/config.py`:

### **Image processing:**
- `IMAGE_CONFIG['image_size']`: 224, 384, hoặc 512
- `IMAGE_CONFIG['fusion_strategy']`: 'early', 'late', 'feature'

### **Data augmentation:**
- `AUGMENTATION_CONFIG['use_augmentation']`: True/False
- Điều chỉnh intensity của từng augmentation

### **Data split:**
- `DATA_SPLIT['train_ratio']`: 0.8 (hoặc 0.7, 0.9)
- `DATA_SPLIT['random_seed']`: 42

### **Model:**
- `MODEL_CONFIG['use_metadata']`: True/False
- `MODEL_CONFIG['architecture']`: 'efficientnet_b3', 'resnet50', 'vit_base_patch16_224'

---

## ⚠️ Common Issues & Solutions

### **Issue 1: Out of Memory**
**Solution:**
- Giảm `batch_size`
- Giảm `image_size` xuống 224
- Giảm `num_workers`
- Sử dụng `mixed_precision=True`

### **Issue 2: Slow data loading**
**Solution:**
- Tăng `num_workers` (4-8)
- Sử dụng SSD thay vì HDD
- Pre-resize images offline

### **Issue 3: Class imbalance không improve**
**Solution:**
- Tăng class weights cho minority classes
- Sử dụng Focal Loss
- Oversample minority classes
- Try different augmentation strengths

### **Issue 4: Validation loss không giảm**
**Solution:**
- Kiểm tra data leakage (train/val overlap)
- Reduce augmentation intensity
- Check label quality
- Try different train/val split

---

## 📚 References

- MILK10k Dataset: https://isic-challenge-data.s3.amazonaws.com/
- Albumentations: https://albumentations.ai/
- Timm models: https://github.com/huggingface/pytorch-image-models
- Multi-label classification: https://scikit-learn.org/stable/modules/multiclass.html

---

**Author:** MILK10k Project Team  
**Last Updated:** 2025-11-30
