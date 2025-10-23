import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
import base64

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | Smart Detection Hub",
    page_icon="🤖",
    layout="wide"
)

st.title("✨ VisionAI Dashboard | Smart Detection Hub")
st.write("Selamat datang di **VisionAI Dashboard** — platform cerdas untuk mendeteksi objek dan menganalisis gambar dengan teknologi **YOLOv8 + CNN**.")

# ================== CACHE MODEL ==================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("best.pt")
    except Exception as e:
        yolo_model = None
        st.warning(f"⚠️ Model YOLO tidak ditemukan atau gagal dimuat: {e}")

    try:
        cnn_model = tf.keras.models.load_model("compressed.h5")
    except Exception as e:
        cnn_model = None
        st.warning(f"⚠️ Model CNN 'compressed.h5' belum ditemukan atau gagal dimuat: {e}")

    return yolo_model, cnn_model

yolo_model, cnn_model = load_models()

# ================== INISIALISASI SESSION STATE ==================
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# ================== FUNGSI CNN ==================
def predict_cnn(image):
    img_resized = image.resize((128, 128))  # ukuran sesuai model CNN kamu
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cnn_model.predict(img_array)
    return prediction

# ================== FUNGSI YOLO ==================
def predict_yolo(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    results = yolo_model.predict(source=img_bytes, conf=0.4, save=False)
    return results[0].plot()[:, :, ::-1]

# ================== FITUR HALAMAN UTAMA ==================
def show_main_page():
    st.markdown("#### 🚀 Unggah Gambar untuk Mulai Analisis")
    uploaded_image = st.file_uploader("Pilih gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.session_state.uploaded_image = uploaded_image
        st.session_state.prediction_done = False
        st.experimental_rerun()

# ================== FITUR HASIL DETEKSI ==================
def show_result_page():
    uploaded_image = st.session_state.uploaded_image
    image = Image.open(uploaded_image)

    st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Deteksi Objek (YOLOv8)")
        if yolo_model:
            yolo_result = predict_yolo(image)
            st.image(yolo_result, caption="Hasil YOLOv8", use_container_width=True)
        else:
            st.error("Model YOLO belum siap.")

    with col2:
        st.subheader("🧠 Prediksi Klasifikasi (CNN)")
        if cnn_model:
            prediction = predict_cnn(image)
            st.write("Output Prediksi:", prediction)
        else:
            st.error("Model CNN belum siap.")

    # Tombol Hapus (tanpa rerun)
    if st.button("🗑️ Hapus Gambar dan Kembali"):
        st.session_state.uploaded_image = None
        st.session_state.prediction_done = False
        show_main_page()

# ================== RENDER HALAMAN ==================
if st.session_state.uploaded_image is None:
    show_main_page()
else:
    show_result_page()
