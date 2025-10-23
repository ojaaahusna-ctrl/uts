import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | Smart Detection & Classification",
    page_icon="👁️",
    layout="wide",
)

# ================== INISIALISASI SESSION STATE ==================
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# ================== MODEL INISIALISASI ==================
@st.cache_resource
def load_models():
    # Load YOLO model
    yolo_model = YOLO("model/yolov8n.pt")  # Ganti jika pakai model YOLO lain

    # Load CNN model
    try:
        cnn_model = tf.keras.models.load_model("model/compressed.h5")
    except Exception as e:
        st.warning("⚠️ Model CNN 'compressed.h5' belum ditemukan atau gagal dimuat.")
        cnn_model = None

    return yolo_model, cnn_model


yolo_model, cnn_model = load_models()

# ================== FUNGSI PENGOLAHAN GAMBAR ==================
IMG_SIZE = (128, 128)  # Ukuran input sesuai model CNN kamu

def predict_cnn(image):
    if cnn_model is None:
        return None, None
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def detect_yolo(image):
    results = yolo_model(image)
    annotated = results[0].plot()
    return annotated, results[0].boxes.data

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Kontrol Dashboard")
mode = st.sidebar.radio("Pilih Mode Input:", ["📸 Kamera", "🖼️ Upload Gambar"])

if st.sidebar.button("Hapus Gambar"):
    st.session_state.uploaded_image = None
    st.session_state.camera_active = False

st.sidebar.markdown("---")
st.sidebar.info("Gunakan **YOLOv8** untuk deteksi objek dan **CNN** untuk klasifikasi gambar.")

# ================== TAMPILAN UTAMA ==================
st.title("👁️ VisionAI Dashboard")
st.markdown("### Smart Detection & Classification System")
st.write(
    """
    Selamat datang di **VisionAI**, dashboard cerdas yang menggabungkan teknologi **YOLOv8**
    untuk deteksi objek dan **Convolutional Neural Network (CNN)** untuk klasifikasi gambar.
    
    Unggah gambar atau gunakan kamera untuk menganalisis objek secara **real-time** 🔍
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input Gambar")

    if mode == "🖼️ Upload Gambar":
        uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.camera_active = False

    elif mode == "📸 Kamera":
        camera_input = st.camera_input("Ambil gambar dengan kamera")
        if camera_input is not None:
            st.session_state.uploaded_image = Image.open(camera_input)
            st.session_state.camera_active = True

    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="Gambar Asli", use_column_width=True)

with col2:
    st.subheader("🔍 Hasil Analisis")

    if st.session_state.uploaded_image is not None:
        image = st.session_state.uploaded_image

        # YOLO Detection
        yolo_annotated, boxes = detect_yolo(np.array(image))
        st.image(yolo_annotated, caption="Deteksi Objek (YOLOv8)", use_column_width=True)

        # CNN Prediction
        predicted_class, confidence = predict_cnn(image)
        if predicted_class is not None:
            st.markdown("### 🧠 Prediksi CNN")
            st.write(f"**Kelas Terdeteksi:** {predicted_class}")
            st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
        else:
            st.info("Model CNN belum dimuat, hanya deteksi YOLO yang aktif.")

    else:
        st.info("Silakan unggah atau ambil gambar terlebih dahulu.")

# ================== FOOTER ==================
st.markdown("---")
st.caption("👁️ **VisionAI Dashboard** | YOLOv8 + CNN | by Ojaa Husna © 2025")
