# 🦴 Bone Region Identification

EfficientNet-B2 tabanlı transfer learning modeli ile X-ray görüntülerinden kemik bölgesi tanıma.

## 🎯 Desteklenen Bölgeler

| Sınıf | Türkçe |
|-------|--------|
| XR_ELBOW | Dirsek |
| XR_FINGER | Parmak |
| XR_FOREARM | Önkol |
| XR_HAND | El |
| XR_HUMERUS | Kol |
| XR_SHOULDER | Omuz |
| XR_WRIST | Bilek |

## 🚀 Demo

👉 [Hugging Face Spaces Demo](https://huggingface.co/spaces/pelinbingl/bone-region-identification)

## 📁 Dataset

[MURA v1.1 — Stanford ML Group](https://www.kaggle.com/datasets/cjinny/mura-v11)
- Train: ~36.800 görüntü
- Val: ~3.200 görüntü
- 7 kemik bölgesi, üst ekstremite X-ray

## 🏗️ Model Mimarisi

- **Backbone:** EfficientNet-B2 (ImageNet pretrained)
- **Classifier:** Linear(1408→512) + SiLU + Linear(512→7)
- **Strateji:** 2 aşamalı transfer learning
  - Aşama 1: Frozen backbone, sadece classifier (5 epoch)
  - Aşama 2: Discriminative LR ile full fine-tune (12 epoch)
- **Optimizer:** AdamW + OneCycleLR
- **Regularization:** Label Smoothing, Dropout, Gradient Clipping, WeightedRandomSampler

## 🔗 İlgili Proje — Kırık Tespiti Pipeline

Bu proje, **Bone Fracture Detection** ile birlikte çalışarak uçtan uca X-ray analizi sağlar:
