# 🦴 Bone X-Ray Analysis Pipeline

EfficientNet tabanlı iki ayrı model ile X-ray görüntülerinden **kemik bölgesi tanıma** ve **kırık tespiti** — uçtan uca medikal AI pipeline'ı.

## 🎯 Pipeline Mimarisi

```
X-Ray Görüntüsü
      ↓
Bone Region Identification  →  Hangi kemik bölgesi? (7 sınıf, %97 accuracy)
      ↓
Bone Fracture Detection     →  Kırık var mı? (%99 accuracy, ROC-AUC: 0.9987)
```

## 🚀 Demo

👉 [Hugging Face Spaces — Canlı Demo](https://huggingface.co/spaces/pelin-bingol/xray-analysis-pipeline)

## 📊 Model Sonuçları

### Bone Region Identification (EfficientNet-B2)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| XR_ELBOW | 0.95 | 0.98 | 0.96 |
| XR_FINGER | 0.98 | 0.96 | 0.97 |
| XR_FOREARM | 0.97 | 0.89 | 0.93 |
| XR_HAND | 0.95 | 0.99 | 0.97 |
| XR_HUMERUS | 0.97 | 0.94 | 0.96 |
| XR_SHOULDER | 0.98 | 1.00 | 0.99 |
| XR_WRIST | 0.98 | 0.98 | 0.98 |
| **Macro Avg** | **0.97** | **0.96** | **0.96** |

### Bone Fracture Detection (EfficientNet-B0)

| Metrik | Değer |
|--------|-------|
| Test Accuracy | %99 |
| ROC-AUC | 0.9987 |
| Precision (fractured) | 1.00 |
| Recall (fractured) | 0.97 |
| F1-Score | 0.99 |

## 🏗️ Teknik Detaylar

### Bone Region Identification
- **Model:** EfficientNet-B2 (ImageNet pretrained)
- **Dataset:** MURA v1.1 (Stanford) — 36,808 train / 3,197 val
- **Strateji:** 2 aşamalı transfer learning
  - Aşama 1: Frozen backbone (5 epoch)
  - Aşama 2: Discriminative LR fine-tune (12 epoch)
- **Teknikler:** WeightedRandomSampler, Label Smoothing, OneCycleLR, Gradient Clipping

### Bone Fracture Detection
- **Model:** EfficientNet-B0 (ImageNet pretrained)
- **Dataset:** Bone Fracture Multi-Region X-ray — 10,581 görüntü
- **Strateji:** 2 aşamalı transfer learning
  - Aşama 1: Frozen backbone (5 epoch)
  - Aşama 2: Full fine-tune (8 epoch)
- **Teknikler:** CLAHE preprocessing, Test-Time Augmentation (TTA)

## 📁 Repo Yapısı

```
Bone_Fracture_Detection/
├── train.py              # Fracture model eğitimi
├── app.py                # Gradio demo (tek model)
├── fracture_model.pth    # Eğitilmiş model
├── confusion_matrix.png
└── training_curves.png

Bone_Region_Identification/
├── train.py              # Region model eğitimi
├── app.py                # İki modeli birleştiren pipeline
├── bone_region_model.pth # Eğitilmiş model
├── confusion_matrix_region.png
└── training_curves_region.png
```

## ⚙️ Kurulum

```bash
git clone https://github.com/pelinbingl/Bone_Region_Identification
cd Bone_Region_Identification
pip install torch torchvision gradio scikit-learn matplotlib Pillow opencv-python-headless numpy
```

### Eğitim
```bash
python train.py
```

### Pipeline Demo
```bash
python combined_app.py
```

## 🔗 İlgili Repolar

| Repo | Açıklama |
|------|----------|
| [Bone_Fracture_Detection](https://github.com/pelinbingl/Bone_Fracture_Detection) | Kırık tespit modeli |
| [Bone_Region_Identification](https://github.com/pelinbingl/Bone_Region_Identification) | Bölge tanıma modeli |

## 👩‍💻 Geliştirici

**Pelin Bingöl**
[LinkedIn](https://linkedin.com/in/pelin-bingöl) • [GitHub](https://github.com/pelinbingl)
