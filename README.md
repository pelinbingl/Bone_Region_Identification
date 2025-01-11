# Kemik Bölgesi Tanıma Projesi

## Genel Bakış

Bu proje, Konvolüyonel Sinir Ağı (CNN) kullanarak röntgen görüntülerini aşağıdaki kemik bölgelerine sınıflandırmayı amaçlar:

- **DIRSEK (ELBOW)**
- **PARMAK (FINGER)**
- **ÖNKOL (FOREARM)**
- **EL (HAND)**
- **KOL KEMİĞİ (HUMERUS)**
- **OMUZ (SHOULDER)**
- **BİLEK (WRIST)**

Model, etiketli veri setleri ile eğitilir ve doğrulama verileri üzerinde test edilir. Çıktılar arasında eğitilmiş bir model ve yeni görüntüler üzerinde tahmin yapmayı sağlayan bir test betiği bulunur.

---

## Gereksinimler

Projenin çalışabilmesi için aşağıdaki kütüphaneler gereklidir:

- **Python 3.7+**
- **TensorFlow 2.x**
- **NumPy**
- **Pillow (PIL)**
- **Matplotlib**
- **OS modülü**

Gerekli kütüphaneleri yüklemek için şu komutu kullanabilirsiniz:

```bash
pip install tensorflow numpy pillow matplotlib
```

---

## Proje Yapısı

```plaintext
Kemik Bölgesi Tanıma/
|-- dataset/
|   |-- train/
|   |   |-- XR_ELBOW/
|   |   |-- XR_FINGER/
|   |   |-- XR_FOREARM/
|   |   |-- XR_HAND/
|   |   |-- XR_HUMERUS/
|   |   |-- XR_SHOULDER/
|   |   |-- XR_WRIST/
|   |-- valid/
|       |-- XR_ELBOW/
|       |-- XR_FINGER/
|       |-- XR_FOREARM/
|       |-- XR_HAND/
|       |-- XR_HUMERUS/
|       |-- XR_SHOULDER/
|       |-- XR_WRIST/
|-- bone_region_model.h5
|-- train_model.py
|-- test_model.py
```

---

## Model Eğitimi

### Betik: `train_model.py`

Bu betik, eğitim veri setini kullanarak CNN modelini eğitir ve doğrulama veri seti üzerinde değerlendirir.

#### Ana Adımlar

1. **Bozuk Görüntüleri Kontrol Etme ve Silme**

```python
def check_and_remove_corrupted_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError):
                print(f"Bozuk görüntü siliniyor: {file_path}")
                os.remove(file_path)
```

2. **Veri Yükleme ve İşleme**

```python
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

3. **Model Mimarisi**

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

4. **Model Eğitimi**

```python
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10
)
```

5. **Modeli Kaydetme**

```python
model.save("bone_region_model.h5")
print("Model başarıyla kaydedildi.")
```

6. **Doğruluk Grafiğini Oluşturma**

Aşağıdaki kod ile modelin eğitim ve doğrulama doğruluğu görselleştirilebilir:

```python
import matplotlib.pyplot as plt

# Eğitim ve doğrulama doğruluğu
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'b', label='Eğitim Doğruluğu')
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok Sayısı')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
```

**Görsel Örnek:** ![image](https://github.com/user-attachments/assets/a148ac02-3482-44f8-b88d-2b68652120a8)


---

## Model Testi

### Betik: `test_model.py`

Bu betik, eğitilmiş modeli yükler ve yeni görüntüler üzerinde tahmin yapar.

---
## Veri Seti

Bu projede kullanılan veri seti, MURA (Musculoskeletal Radiographs) veri setidir. Bu veri setine şu bağlantıdan erişebilirsiniz:

[https://www.kaggle.com/code/azaemon/mura-classification/input](https://www.kaggle.com/code/azaemon/mura-classification/input)

---
#### Ana Adımlar

1. **Modeli Yükleme**

```python
model = load_model("bone_region_model.h5")
print("Model başarıyla yüklendi.")
```

2. **Görüntüyü İşleme**

```python
img = Image.open(test_image_path)
if img.mode != "RGB":
    img = img.convert("RGB")

img = img.resize((224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0
```

3. **Tahmin Yapma**

```python
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_names[predicted_class]

print(f"Görüntü: {test_image_path}")
print(f"Tahmin Edilen Sınıf İndeksi: {predicted_class}")
print(f"Tahmin Edilen Sınıf Adı: {predicted_class_name}")
```

#### Örnek Çıktı

![image1](https://github.com/user-attachments/assets/3458186d-fabd-4ad6-9819-1ce4faab848d)

```plaintext
Model başarıyla yüklendi.
Görüntü: ![image1](https://github.com/user-attachments/assets/039342ad-8c5e-4812-a711-3dbaed45a825)

Tahmin Edilen Sınıf İndeksi: 3
Tahmin Edilen Sınıf Adı: HAND
```

---

## Notlar

- Veri setinin `train` ve `valid` dizin yapısına uygun olduğundan emin olun.
- Bozuk görüntüler otomatik olarak kaldırılır.
- `test_model.py` betiğindeki `class_names` değişkenini veri seti etiketleriyle uyumlu olacak şekilde güncelleyin.

---

## Lisans

Bu proje yalnızca eğitim amaçlıdır.




