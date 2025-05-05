import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests

# Fungsi untuk memuat gambar dari URL
def load_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

# Ganti dengan URL gambar yang ingin Anda deteksi
image_url = "haikal.jpeg"
image = load_image(image_url)

# Muat model dan processor
model_name = "google/vit-base-patch16-224-in21k"  # Ganti dengan model yang Anda inginkan
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Proses gambar
inputs = image_processor(images=image, return_tensors="pt")

# Lakukan prediksi
with torch.no_grad():
    outputs = model(**inputs)

# Ambil prediksi
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# Ambil nama kelas
labels = model.config.id2label
predicted_label = labels[predicted_class_idx]

print(f"Predicted label: {predicted_label}")
