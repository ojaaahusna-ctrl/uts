import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import base64
import requests
import re

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | Final Version",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== INITIALIZE SESSION STATE ==================
defaults = {
    'page': 'home',
    'selected_image_bytes': None,
    'cnn_conf': 0.85
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================== STYLE KUSTOM (CSS) ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Playfair+Display:wght@700&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #E6FFFA 0%, #B2F5EA 100%);
    color: #2D3748; 
}
.stApp, .main, [data-testid="stSidebar"] {
    background: linear-gradient(135deg, #E6FFFA 0%, #B2F5EA 100%); 
    color: #2D3748;
}
h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText, 
[data-testid="stMarkdownContainer"],
.stRadio > label,
[data-testid="stMetricLabel"], 
[data-testid="stMetricValue"], 
[data-testid="stAlert"]
{
    color: #2D3748 !important; 
}
#home-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    width: 100%;
    margin-top: 1rem;
}
#home-container > div { max-width: 800px; width: 100%; }
[data-testid="stSidebar"] { background-color: #F0FFF4; }

.header {
    background-color: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.8);
    margin-bottom: 2rem; 
}
.header h1 {
    font-family: 'Playfair Display', serif;
    color: #2D3748; 
    font-size: 3rem;
}
.menu-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 2rem 1.5rem;
    border-radius: 15px;
    text-align: center;
    transition: all 0.3s ease-in-out;
    height: 100%;
}
.stButton>button {
    background-color: #319795;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}
.stButton>button:hover { background-color: #2C7A7B; }
div[data-baseweb="input"], div[data-baseweb="textarea"] {
    background-color: #FFFFFF !important;
    border-radius: 8px;
    border: 1px solid #B2F5EA;
    color: #2D3748 !important;
}
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 2px dashed #B2F5EA !important;
}
[data-testid="stFileUploader"] section * {
    color: #2D3748 !important; 
}
.stFileUploader > div > label {
    color: #2D3748 !important; 
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource(show_spinner="📦 Memuat model YOLO...")
def load_yolo_model():
    try:
        model = YOLO("model/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        return None

@st.cache_resource(show_spinner="📦 Memuat model CNN...")
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("model/compressed.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== UTILITAS ==================
def clear_image_state():
    st.session_state['selected_image_bytes'] = None

def go_home():
    st.session_state.page = 'home'
    clear_image_state()
    st.rerun()

def reset_and_rerun():
    clear_image_state()
    current_page = st.session_state.get('page', 'home')
    if current_page in ['yolo', 'cnn']:
        st.rerun()

# ================== HALAMAN HOME ==================
def home_page():
    st.markdown('<div id="home-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="header">
        <h1>✨ VisionAI Dashboard ✨</h1>
        <p>Platform Interaktif untuk Deteksi & Klasifikasi Gambar</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Pilih Tugas yang Ingin Dilakukan:", anchor=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="menu-card"><h3>🌭 Deteksi Objek</h3><p>Gunakan model YOLO untuk mendeteksi Hotdog vs Not-Hotdog.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Deteksi", use_container_width=True, key="yolo_nav"):
            st.session_state.page = 'yolo'
            clear_image_state()
            st.rerun()
    with col2:
        st.markdown('<div class="menu-card"><h3>🐆 Klasifikasi Gambar</h3><p>Gunakan model CNN untuk mengklasifikasikan Cheetah dan Hyena.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Klasifikasi", use_container_width=True, key="cnn_nav"):
            st.session_state.page = 'cnn'
            clear_image_state()
            st.rerun()

    st.markdown("---")
    st.info("Proyek ini dibuat oleh **Raudhatul Husna** sebagai bagian dari Ujian Tengah Semester.", icon="🎓")
    st.markdown('</div>', unsafe_allow_html=True)

# ================== HALAMAN MODEL ==================
def run_model_page(page_type):
    if page_type == 'yolo':
        title = "🌭 Deteksi Objek: Hotdog vs Not-Hotdog"
        model_loader = load_yolo_model
        button_text = "🔍 Mulai Deteksi"
    else:
        title = "🐆 Klasifikasi Gambar: Cheetah vs Hyena"
        model_loader = load_cnn_model
        button_text = "🔮 Lakukan Prediksi"

    st.button("⬅️ Kembali ke Menu Utama", on_click=go_home)
    st.header(title)

    if page_type == 'cnn':
        st.info("💡 Model ini hanya mengenali **Cheetah** dan **Hyena**. Gunakan slider di sidebar untuk atur ambang keyakinan.", icon="💡")
    if page_type == 'yolo':
        st.info("⚠️ Model ini hanya dilatih untuk mendeteksi **Hotdog**.", icon="🍔")

    model = model_loader()
    if not model:
        st.stop()

    image_bytes = None
    source_key = f"{page_type}_source"
    upload_key = f"{page_type}_upload"
    url_key = f"{page_type}_url_input"
    cam_key = f"{page_type}_cam"

    with st.sidebar:
        st.title("⚙️ Pengaturan")
        st.markdown("---")

        if page_type == 'cnn':
            st.session_state.cnn_conf = st.slider("Min. Keyakinan (CNN)", 0.0, 1.0, st.session_state.cnn_conf, 0.05)
            st.warning(f"Hasil di bawah {st.session_state.cnn_conf:.0%} akan ditolak.", icon="⚖️")

        if page_type == 'yolo':
            confidence_threshold = st.slider("Tingkat Keyakinan (YOLO)", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")

        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera", "🔗 Input URL Gambar"], key=source_key)

        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=upload_key)
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
                st.session_state['selected_image_bytes'] = image_bytes

        elif source_choice == "📸 Ambil dari Kamera":
            camera_input = st.camera_input("Arahkan kamera", key=cam_key)
            if camera_input:
                image_bytes = camera_input.getvalue()
                st.session_state['selected_image_bytes'] = image_bytes
            st.info("⚠️ Kamera hanya berfungsi di koneksi HTTPS.", icon="🛡️")

        elif source_choice == "🔗 Input URL Gambar":
            url = st.text_input("Masukkan URL Gambar:", value=st.session_state.get(url_key, ''), key=url_key)
            if url:
                if not re.match(r'^https?://[^\s]+$', url):
                    st.error("❌ URL tidak valid.", icon="⚠️")
                else:
                    try:
                        with st.spinner("Mengunduh gambar..."):
                            response = requests.get(url, timeout=10)
                            response.raise_for_status()
                            if 'image' not in response.headers.get('Content-Type', '').lower():
                                st.error("❌ URL bukan file gambar valid.", icon="⚠️")
                            else:
                                image_bytes = response.content
                                st.session_state['selected_image_bytes'] = image_bytes
                                st.success("✅ Gambar berhasil diunduh.", icon="🌐")
                    except Exception as e:
                        st.error(f"Gagal mengunduh gambar: {e}")

    if image_bytes is None:
        image_bytes = st.session_state.get('selected_image_bytes')

    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            st.error("❌ Format gambar tidak didukung.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
            st.button("🗑️ Hapus Gambar & Reset", use_container_width=True, key=f"{page_type}_reset", on_click=reset_and_rerun)

        placeholder = col2.empty()
        placeholder.info("Tekan tombol di bawah untuk memproses gambar.")

        if st.button(button_text, use_container_width=True, key=f"{page_type}_predict"):
            with st.spinner("🧠 Menganalisis gambar..."):
                if page_type == 'yolo':
                    results = model(image, conf=confidence_threshold)
                    plot_result = results[0].plot()
                    if plot_result is not None:
                        result_img_rgb = cv2.cvtColor(plot_result, cv2.COLOR_BGR2RGB)
                        with placeholder.container():
                            st.subheader("🎯 Hasil Deteksi")
                            st.image(result_img_rgb, use_container_width=True)
                            boxes = results[0].boxes
                            if len(boxes) > 0:
                                for i, box in enumerate(boxes):
                                    class_name = model.names[int(box.cls)] if hasattr(model, "names") else "Objek"
                                    st.success(f"Objek {i+1}: `{class_name}` | Keyakinan: `{box.conf[0]:.2%}`")
                            else:
                                st.success("✅ Tidak ditemukan objek 'Hotdog' → **Not-Hotdog**", icon="👍")
                    else:
                        st.warning("Tidak ada hasil deteksi.")
                else:
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    input_shape = model.input_shape[1:3]
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0)
                    preds_output = model.predict(img_array, verbose=0)[0]

                    if len(preds_output) == 1:
                        prob = preds_output[0]
                        pred_idx = 1 if prob > 0.5 else 0
                        pred_prob = max(prob, 1-prob)
                        preds_for_display = [1-prob, prob]
                    else:
                        pred_idx = np.argmax(preds_output)
                        pred_prob = np.max(preds_output)
                        preds_for_display = preds_output

                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        if pred_prob >= st.session_state.cnn_conf:
                            st.metric("Prediksi:", CLASS_NAMES_CNN[pred_idx])
                            st.metric("Keyakinan:", f"{pred_prob:.2%}")
                            st.success(f"Gambar terdeteksi sebagai {CLASS_NAMES_CNN[pred_idx]}.", icon="✅")
                            st.subheader("📊 Distribusi Probabilitas")
                            for i, p in enumerate(preds_for_display):
                                st.write(f"{CLASS_NAMES_CNN[i]}: {p:.2%}")
                                st.progress(float(p))
                        else:
                            st.error("❌ Gambar Tidak Terdeteksi", icon="🚫")
                            st.warning(f"Keyakinan tertinggi ({pred_prob:.2%}) di bawah ambang batas ({st.session_state.cnn_conf:.2%}).")

# ================== ROUTER UTAMA ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')
