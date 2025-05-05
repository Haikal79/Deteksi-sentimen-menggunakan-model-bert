# Install dependensi
!pip install -q transformers pytesseract pillow torch
!apt-get -qq install -y tesseract-ocr

# Import library
from PIL import Image
import pytesseract
from transformers import pipeline
from google.colab import files
import IPython.display as display
from IPython.display import HTML
import os

# Fungsi untuk memberi warna dan emoji pada hasil
def display_sentiment(label, score):
    label = label.lower()
    emoji = ""
    color = "#ccc"
    if "5" in label:
        emoji = "😄"
        color = "#a0e3a0"
    elif "4" in label:
        emoji = "😊"
        color = "#d4f1c5"
    elif "3" in label:
        emoji = "😐"
        color = "#fef3b8"
    elif "2" in label:
        emoji = "😕"
        color = "#fbd4b4"
    elif "1" in label:
        emoji = "😠"
        color = "#f8a3a3"

    html = f"""
    <div style='
        background-color: {color};
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        width: fit-content;
        margin-top: 10px;
    '>
        <strong>Label:</strong> {label.upper()} {emoji}<br>
        <strong>Confidence:</strong> {score:.4f}
    </div>
    """
    display.display(HTML(html))

# Upload gambar
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path)

# Tampilkan gambar
print("🖼️ Gambar yang diunggah:")
display.display(image)

# Ekstrak teks
text = pytesseract.image_to_string(image).strip()

if not text:
    print("⚠️ Tidak ada teks terdeteksi.")
else:
    print("\n📝 Teks dari gambar:")
    print(text)

    # Analisis sentimen
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    result = sentiment_pipeline(text)[0]

    print("\n🔍 Hasil Analisis Sentimen:")
    display_sentiment(result['label'], result['score'])
