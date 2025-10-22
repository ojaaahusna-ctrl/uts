import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import base64

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | Final Version",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== INITIALIZE SESSION STATE ==================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ================== DATA CONTOH GAMBAR ==================
CHEETAH_B64 = "/9j/4AAQSkZJRgABAQAAQABAAD..."
HYENA_B64 = "/9j/4AAQSkZJRgABAQAAQABAAD..."
HOTDOG_B64 = "/9j/4AAQSkZJRgABAQAAQABAAD..."

# ================== STYLE KUSTOM ==================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg,#E6FFFA 0%,#B2F5EA 100%);}
[data-testid="stSidebar"] {background-color:#F0FFF4;}
.header {background-color:rgba(255,255,255,0.5);backdrop-filter:blur(10px);padding:2.5rem;
border-radius:20px;text-align:center;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.8);}
.header h1{font-family:'Playfair Display',serif;color:#2D3748;font-size:3rem;}
.header p{color:#4A5568;font-size:1.2rem;}
.menu-card{background:#fff;border:1px solid #E2E8F0;padding:2rem 1.5rem;border-radius:15px;text-align:center;
transition:all .3s;height:100%;}
.menu-card:hover{transform:translateY(-8px);box-shadow:0 8px 30px rgba(49,151,149,0.15);border-color:#319795;}
.menu-card h3{color:#2C7A7B;font-family:'Playfair Display',serif;}
.menu-card p{color:#4A5568;}
.stButton>button{background:#319795;color:white;border-radius:10px;border:none;padding:10px 20px;font-weight:bold;}
.stButton>button:hover{background:#2C7A7B;}
h1,h2,h3,h4,h5,h6,p,li,label{color:#2D3748 !important;}
</style>
""", unsafe_allow_html=True)

# ================== CACHE MODEL ==================
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("model/best.pt")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        return None

@st.cache_resource
def load_cnn_model():
    try:
        return tf.keras.models.load_model("model/compressed.h5", compile=True)
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== HALAMAN HOME ==================
def home_page():
    st.markdown("""
    <div class="header">
        <h1>✨ VisionAI Dashboard ✨</h1>
        <p>Platform Interaktif untuk Deteksi & Klasifikasi Gambar</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Pilih Tugas yang Ingin Dilakukan:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="menu-card"><h3>🌭 Deteksi Objek</h3><p>Gunakan model YOLO untuk mendeteksi Hotdog vs Not-Hotdog.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Deteksi", use_container_width=True, key="yolo_nav"):
            st.session_state.page = 'yolo'
            st.rerun()

    with col2:
        st.markdown('<div class="menu-card"><h3>🐆 Klasifikasi Gambar</h3><p>Gunakan model CNN untuk mengklasifikasikan Cheetah dan Hyena.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Klasifikasi", use_container_width=True, key="cnn_nav"):
            st.session_state.page = 'cnn'
            st.rerun()

    st.markdown("---")
    st.info("Proyek ini dibuat oleh **Balqis Isaura** sebagai bagian dari Ujian Tengah Semester.", icon="🎓")

# ================== HALAMAN MODEL ==================
def run_model_page(page_type):
    if page_type == 'yolo':
        title = "🌭 Deteksi Objek: Hotdog vs Not-Hotdog"
        model_loader = load_yolo_model
        sample_images = {"Contoh Hotdog": HOTDOG_B64}
        button_text = "🔍 Mulai Deteksi"
    else:
        title = "🐆 Klasifikasi Gambar: Cheetah vs Hyena"
        model_loader = load_cnn_model
        sample_images = {"Contoh Cheetah": CHEETAH_B64, "Contoh Hyena": HYENA_B64}
        button_text = "🔮 Lakukan Prediksi"

    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        st.rerun()

    st.header(title)
    model = model_loader()
    if not model:
        return

    image_bytes = None

    with st.sidebar:
        st.title("⚙️ Pengaturan")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera", "🖼️ Pilih Contoh"], key=f"{page_type}_source")

        if page_type == 'yolo':
            st.markdown("---")
            confidence_threshold = st.slider("Tingkat Keyakinan", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")

        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"{page_type}_upload")
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
        elif source_choice == "📸 Ambil dari Kamera":
            camera_input = st.camera_input("Arahkan kamera", key=f"{page_type}_cam")
            if camera_input:
                image_bytes = camera_input.getvalue()
        else:
            st.subheader("Pilih gambar dari galeri:")
            cols = st.columns(len(sample_images))
            for idx, (caption, b64_string) in enumerate(sample_images.items()):
                with cols[idx]:
                    st.image(f"data:image/jpeg;base64,{b64_string}", caption=caption, use_container_width=True)
                    if st.button(f"Gunakan {caption}", key=f"sample_{idx}", use_container_width=True):
                        image_bytes = base64.b64decode(b64_string)

    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
        placeholder = col2.empty()
        placeholder.info("Hasil akan muncul di sini setelah diproses.")

        if st.button(button_text, type="primary", use_container_width=True):
            with st.spinner("🧠 Menganalisis gambar..."):
                if page_type == 'yolo':
                    results = model(image, conf=confidence_threshold)
                    result_img_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    with placeholder.container():
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img_rgb, use_container_width=True)
                        st.subheader("📋 Detail Deteksi")
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            for i, box in enumerate(boxes):
                                st.success(f"**Objek {i+1}:** {model.names[int(box.cls)]} | **Keyakinan:** {box.conf[0]:.2%}", icon="✅")
                        else:
                            st.warning("Tidak ada objek terdeteksi.", icon="⚠️")
                else:
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    input_shape = model.input_shape[1:3]
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0)
                    preds = model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(preds)
                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Prediksi Utama:", CLASS_NAMES_CNN.get(pred_idx))
                        st.metric("Tingkat keyakinan:", f"{preds[pred_idx]:.2%}")
                        st.subheader("📊 Distribusi Probabilitas")
                        for i, prob in enumerate(preds):
                            st.progress(float(prob), text=f"{CLASS_NAMES_CNN.get(i)}: {prob:.2%}")

# ================== ROUTER ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')
