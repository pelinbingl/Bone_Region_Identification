import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Veri yolları
test_dir = "D:\\Project\\Bone Region Identification\\dataset\\valid\\XR_ELBOW\\patient11186\\image1.png"  # Test veri seti yolu
model_path = "D:\\Project\\Bone Region Identification\\bone_region_model.h5"  # Kaydedilen model yolu

# Görüntü boyutları ve batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Kaydedilen modeli yükleme
model = load_model(model_path)
print("Model başarıyla yüklendi.")

# 2. Test verilerini yükleme ve ön işleme
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Tahminlerin sırasını korumak için karıştırmayı kapat
)

# 3. Modeli test etme
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Seti Doğruluğu: {test_accuracy * 100:.2f}%")

# 4. Tahminler yapma ve sınıf isimlerini görüntüleme
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Sınıf etiketlerini yükleme (Eğer sınıf etiketlerini kaydettiysen)
class_indices = test_generator.class_indices
classes = {v: k for k, v in class_indices.items()}  # Ters çevirerek sınıf isimlerini indekslere eşle

# Tahminlerin detaylı çıktısını göster
filenames = test_generator.filenames
for i in range(len(filenames)):
    print(f"Dosya: {filenames[i]}, Tahmin Edilen Sınıf: {classes[predicted_classes[i]]}")
