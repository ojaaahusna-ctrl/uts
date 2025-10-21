import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="YOLO + Keras Dashboard", layout="wide")

st.title("🚀 Deteksi & Klasifikasi dengan YOLO (.pt) dan Keras (.h5)")

@st.cache_resource
def load_models():
    # Muat model YOLO (.pt)
    yolo_model = YOLO("model/best.pt")
    
    # Muat model klasifikasi (.h5)
    classifier = tf.keras.models.load_model("model/Raudhatul Husna_laporan2.h5")
    
    return yolo_model, classifier

# Load kedua model
yolo_model, classifier = load_models()

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    st.image(image, caption="Gambar Asli", use_column_width=True)
    
    # Deteksi objek dengan YOLO
    st.subheader("🔍 Deteksi Objek (YOLO)")
    results = yolo_model.predict(img_array)
    result_img = results[0].plot()  # Gambar hasil deteksi
    
    st.image(result_img, caption="Hasil Deteksi YOLO", use_column_width=True)
    
    # Klasifikasi dengan model H5
    st.subheader("🧠 Klasifikasi (Keras .h5)")
    resized = cv2.resize(img_array, (224, 224)) / 255.0
    reshaped = np.expand_dims(resized, axis=0)
    pred = classifier.predict(reshaped)
    
    predicted_class = np.argmax(pred, axis=1)[0]
    st.success(f"Prediksi kelas: {predicted_class}")
