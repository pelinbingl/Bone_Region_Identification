#combined_app.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
import gradio as gr

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sınıf isimleri ───────────────────────────────────────
REGION_CLASSES = {
    0: "🦾 Dirsek (Elbow)",
    1: "🖐️ Parmak (Finger)",
    2: "💪 Önkol (Forearm)",
    3: "✋ El (Hand)",
    4: "🦵 Kol (Humerus)",
    5: "🏋️ Omuz (Shoulder)",
    6: "⌚ Bilek (Wrist)",
}
FRACTURE_CLASSES = {
    0: "🔴 Kırık (Fractured)",
    1: "🟢 Normal (Not Fractured)",
}

# ── Model yükleme ─────────────────────────────────────────
def load_region_model():
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    )
    model.load_state_dict(torch.load(
        r"D:\Projeler\Python\Bone_Region_Identification\bone_region_model.pth",
        map_location=DEVICE
    ))
    model.eval()
    return model.to(DEVICE)

def load_fracture_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load(
        r"D:\Projeler\Python\Bone_Fracture_Detection\fracture_model.pth",
        map_location=DEVICE
    ))
    model.eval()
    return model.to(DEVICE)

print("Modeller yükleniyor...")
region_model   = load_region_model()
fracture_model = load_fracture_model()
print("Hazır!")

# ── Transform ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Tahmin fonksiyonu ─────────────────────────────────────
def predict(image):
    if image is None:
        return "⚠️ Lütfen bir görüntü yükleyin.", {}, {}

    img    = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Bölge tahmini
        region_out   = region_model(tensor)
        region_probs = torch.softmax(region_out, dim=1)[0]
        region_idx   = region_probs.argmax().item()
        region_conf  = float(region_probs[region_idx]) * 100

        # Kırık tahmini
        frac_out   = fracture_model(tensor)
        frac_probs = torch.softmax(frac_out, dim=1)[0]
        frac_idx   = frac_probs.argmax().item()
        frac_conf  = float(frac_probs[frac_idx]) * 100

    region_name  = REGION_CLASSES[region_idx]
    frac_name    = FRACTURE_CLASSES[frac_idx]

    # Sonuç kartı
    durum_renk = "🔴" if frac_idx == 0 else "🟢"
    sonuc = f"""
## {durum_renk} Analiz Sonucu

| | Sonuç | Güven |
|---|---|---|
| 📍 Kemik Bölgesi | {region_name} | %{region_conf:.1f} |
| 🦴 Kırık Durumu | {frac_name} | %{frac_conf:.1f} |

---
⚕️ *Bu uygulama yalnızca eğitim amaçlıdır. Tıbbi teşhis için hekime başvurunuz.*
"""

    region_dist = {REGION_CLASSES[i]: float(region_probs[i]) for i in range(7)}
    frac_dist   = {FRACTURE_CLASSES[i]: float(frac_probs[i]) for i in range(2)}

    return sonuc, region_dist, frac_dist

# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="🦴 Bone X-Ray AI") as demo:

    gr.Markdown("""
    # 🦴 Bone X-Ray Analysis Pipeline
    ### Kemik Bölgesi Tanıma + Kırık Tespiti
    
    X-ray görüntüsünü yükle → **hangi kemik bölgesi** ve **kırık var mı** öğren.
    
    > `Bölge Modeli: EfficientNet-B2 (MURA, 7 sınıf)`  
    > `Kırık Modeli: EfficientNet-B0 — Test Accuracy: %99 | ROC-AUC: 0.9987`
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📂 X-Ray Görüntüsü Yükle")
            analyze_btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")
            gr.Markdown("**Örnek X-ray görüntüsü yükleyerek test edebilirsiniz.**")

        with gr.Column(scale=1):
            result_md      = gr.Markdown(label="Sonuç")
            region_label   = gr.Label(num_top_classes=7, label="📍 Kemik Bölgesi Dağılımı")
            fracture_label = gr.Label(num_top_classes=2, label="🦴 Kırık Durumu")

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_md, region_label, fracture_label]
    )

    gr.Markdown("""
    ---
    ### 🔗 Projeler
    | Proje | Açıklama |
    |---|---|
    | [Bone Fracture Detection](https://github.com/pelinbingl/Bone_Fracture_Detection) | EfficientNet-B0, %99 accuracy |
    | [Bone Region Identification](https://github.com/pelinbingl/Bone_Region_Identification) | EfficientNet-B2, 7 sınıf |
    
    **Geliştirici:** [Pelin Bingöl](https://linkedin.com/in/pelin-bingöl)
    """)

if __name__ == "__main__":
    demo.launch(share=False)