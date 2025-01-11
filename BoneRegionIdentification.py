import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

# Veri seti yolunu belirtin
dataset_path = "MURA-v1.1"  # İndirdiğiniz veri setinin yolu

# Görüntü boyutları
IMG_SIZE = 224  # Resim boyutunu küçültmek için
BATCH_SIZE = 32

# 1. Görüntüleri yükleme ve etiketleme
def load_images_and_labels(dataset_path):
    images = []
    labels = []
    for class_name in ["positive", "negative"]:  # Hastalık varsa "positive", yoksa "negative"
        class_path = os.path.join(dataset_path, class_name)
        label = 1 if class_name == "positive" else 0
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            try:
                img = Image.open(file_path).convert("L")  # Siyah beyaz
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
    return np.array(images), np.array(labels)

images, labels = load_images_and_labels(dataset_path)
print(f"Toplam görüntü sayısı: {len(images)}")

# 2. Veriyi normalleştirme
images = images / 255.0  # Piksel değerlerini [0, 1] aralığına dönüştür

# 3. Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 4. Modeli Oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # İkili sınıflandırma
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Modeli Eğitme
history = model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_split=0.2)

# 6. Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_accuracy * 100:.2f}%")

# 7. Eğitim ve doğrulama doğruluğunu görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()