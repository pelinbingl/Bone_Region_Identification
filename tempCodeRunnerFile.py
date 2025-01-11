import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Test görüntüsü ve model yolu
test_image_path = "D:\\Project\\Bone Region Identification\\dataset\\valid\\XR_HAND\\patient11190\\study1_negative\\image1.png"
model_path = "D:\\Project\\Bone Region Identification\\bone_region_model.h5"

# Görüntü boyutları
IMG_SIZE = (224, 224)

# Sınıf isimleri (Dizin adlarını kullanarak)
class_names = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]

# 1. Kaydedilen modeli yükleme
model = load_model(model_path)
print("Model başarıyla yüklendi.")

# 2. Test görüntüsünü yükleme ve ön işleme
try:
    img = Image.open(test_image_path)
    
    # Görüntüyü RGB formatına dönüştür
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(IMG_SIZE)  # Görüntüyü modelin giriş boyutuna yeniden boyutlandır
    img_array = img_to_array(img)  # Görüntüyü diziye çevir
    img_array = np.expand_dims(img_array, axis=0)  # Model için batch boyutunu ekle
    img_array = img_array / 255.0  # Normalizasyon (0-1 arasında değerler)

    # 3. Tahmin yapma
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # 4. Tahmin edilen sınıf adı
    predicted_class_name = class_names[predicted_class]

    print(f"Görüntü: {test_image_path}")
    print(f"Tahmin Edilen Sınıf İndeksi: {predicted_class}")
    print(f"Tahmin Edilen Sınıf Adı: {predicted_class_name}")
except Exception as e:
    print(f"Görüntü işleme hatası: {e}")
