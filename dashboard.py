import streamlit as st
from ultranalytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | Deteksi & Klasifikasi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== INITIALIZE SESSION STATE ==================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ================== STYLE KUSTOM (CSS) ==================
# CSS diperbarui untuk tampilan visual yang lebih menarik
st.markdown("""
<style>
    /* Mengubah font utama */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Kustomisasi header */
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #4a4e69;
    }
    .header h1 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .header p {
        color: #c0c0d0;
        font-size: 1.1rem;
    }
    
    /* Style untuk kartu menu di halaman utama */
    .menu-card {
        padding: 2rem 1.5rem;
        border-radius: 10px;
        background-color: #161b22;
        border: 1px solid #30363d;
        text-align: center;
        transition: all 0.3s ease-in-out;
        height: 100%;
    }
    .menu-card:hover {
        border-color: #4a4e69;
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .menu-card h3 {
        color: #e0e0e0;
        margin-bottom: 1rem;
    }
    .menu-card p {
        color: #b0b0d0;
        font-size: 0.9rem;
    }

    /* Style untuk kartu hasil prediksi yang lebih menarik */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #1a1a2e;
        border: 1px solid #4a4e69;
        margin-top: 1rem;
    }
    .result-card .main-prediction {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00aaff;
        text-align: center;
    }
    .result-card .confidence {
        font-size: 1.1rem;
        color: #b0b0d0;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-card h4 {
        color: #e0e0e0;
        border-bottom: 2px solid #4a4e69;
        padding-bottom: 0.5rem;
        margin-top: 0;
    }

    /* Bar Chart Kustom untuk Probabilitas (menggantikan st.progress) */
    .probability-chart {
        width: 100%;
        margin-top: 1rem;
    }
    .bar-wrapper {
        margin-bottom: 0.8rem;
    }
    .bar-label {
        color: #e0e0e0;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        display: flex;
        justify-content: space-between;
    }
    .bar-container {
        width: 100%;
        background-color: #30363d;
        border-radius: 5px;
        height: 22px;
        overflow: hidden;
    }
    .bar-fill {
        background: linear-gradient(90deg, #007bff 0%, #00aaff 100%);
        height: 100%;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
        color: white;
        font-size: 0.8rem;
        font-weight: 500;
        transition: width 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ================== CACHE MODEL ==================
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("model/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        return None

@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("model/compressed.h5", compile=True)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== HELPER FUNCTIONS ==================
def render_probability_chart(predictions, class_names):
    """Merender bar chart kustom HTML untuk probabilitas."""
    chart_html = '<div class="probability-chart">'
    for i, prob in enumerate(predictions[0]):
        class_name = class_names.get(i, f"Kelas {i}")
        percentage = f"{prob:.1%}"
        chart_html += f"""
        <div class="bar-wrapper">
            <div class="bar-label">
                <span>{class_name}</span>
                <span>{percentage}</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill" style="width: {prob*100}%;"></div>
            </div>
        </div>
        """
    chart_html += '</div>'
    st.markdown(chart_html, unsafe_allow_html=True)


# ================== FUNGSI UNTUK SETIAP HALAMAN ==================
def home_page():
    st.markdown("""
    <div class="header">
        <h1>🤖 VisionAI Dashboard</h1>
        <p>Platform Interaktif untuk Deteksi dan Klasifikasi Gambar</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Pilih Tugas yang Ingin Dilakukan:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="menu-card"><h3>🌭 Deteksi Objek</h3><p>Gunakan model YOLO untuk mendeteksi Hotdog vs Not-Hotdog dalam sebuah gambar.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Deteksi", use_container_width=True, key="yolo_nav"):
            st.session_state.page = 'yolo'
            st.rerun()
    with col2:
        st.markdown('<div class="menu-card"><h3>🐆 Klasifikasi Gambar</h3><p>Gunakan model CNN untuk mengklasifikasikan gambar antara Cheetah dan Hyena.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Klasifikasi", use_container_width=True, key="cnn_nav"):
            st.session_state.page = 'cnn'
            st.rerun()
    st.markdown("---")
    st.info("Proyek ini dibuat oleh **Balqis Isaura** sebagai bagian dari Ujian Tengah Semester.", icon="ℹ️")

def yolo_page():
    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        st.rerun()
    st.header("🌭 Deteksi Objek: Hotdog vs Not-Hotdog")
    with st.sidebar:
        st.image("https://i.imgur.com/G4f4bJb.png", width=100)
        st.title("⚙️ Pengaturan Deteksi")
        st.markdown("---")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera"], key="yolo_source")
        confidence_threshold = st.slider("Tingkat Keyakinan", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")
    yolo_model = load_yolo_model()
    if not yolo_model: return
    image_bytes = None
    if source_choice == "📤 Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="yolo_upload")
        if uploaded_file: image_bytes = uploaded_file.getvalue()
    else:
        camera_input = st.camera_input("Arahkan kamera", key="yolo_cam")
        if camera_input: image_bytes = camera_input.getvalue()
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("🎯 Hasil Deteksi")
            # Placeholder untuk hasil
            result_placeholder = st.empty()
            result_placeholder.info("Hasil deteksi akan muncul di sini setelah diproses.")
        if st.button("🔍 Mulai Deteksi", type="primary", use_container_width=True):
            with st.spinner("🧠 Menganalisis gambar..."):
                results = yolo_model(image, conf=confidence_threshold)
                result_img = results[0].plot()
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                with result_placeholder.container():
                    st.image(result_img_rgb, use_column_width=True)
            st.markdown("---")
            st.subheader("📋 Detail Deteksi")
            with st.container(border=True):
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        class_name = yolo_model.names[int(box.cls)]
                        confidence = box.conf[0]
                        st.write(f"**Objek {i+1}:** `{class_name}` | **Keyakinan:** `{confidence:.2%}`")
                else:
                    st.success("Tidak ada objek yang terdeteksi.", icon="✅")

def cnn_page():
    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        st.rerun()
    st.header("🐆 Klasifikasi Gambar: Cheetah vs Hyena")
    with st.sidebar:
        st.image("https://i.imgur.com/G4f4bJb.png", width=100)
        st.title("⚙️ Pengaturan Klasifikasi")
        st.markdown("---")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera"], key="cnn_source")
    cnn_model = load_cnn_model()
    if not cnn_model: return
    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
    image_bytes = None
    if source_choice == "📤 Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="cnn_upload")
        if uploaded_file: image_bytes = uploaded_file.getvalue()
    else:
        camera_input = st.camera_input("Arahkan kamera", key="cnn_cam")
        if camera_input: image_bytes = camera_input.getvalue()
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("🎯 Hasil Prediksi")
            result_placeholder = st.empty()
            with result_placeholder.container():
                 st.info("Hasil prediksi akan muncul di sini.")
        if st.button("🔮 Lakukan Prediksi", type="primary", use_container_width=True):
            with st.spinner("🧠 Memproses dan memprediksi..."):
                input_shape = cnn_model.input_shape[1:3]
                img_resized = image.resize(input_shape)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                predictions = cnn_model.predict(img_array, verbose=0)
                confidence = float(np.max(predictions))
                predicted_class_idx = int(np.argmax(predictions))
                predicted_class_name = CLASS_NAMES_CNN.get(predicted_class_idx, f"Kelas {predicted_class_idx}")
                
                # Tampilkan hasil di placeholder
                with result_placeholder.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>Hasil Utama</h4>
                        <p class="main-prediction">{predicted_class_name}</p>
                        <p class="confidence">Keyakinan: {confidence:.2%}</p>
                        <h4>Distribusi Probabilitas</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    # Render bar chart kustom
                    render_probability_chart(predictions, CLASS_NAMES_CNN)


# ================== ROUTER UTAMA APLIKASI ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    yolo_page()
elif st.session_state.page == 'cnn':
    cnn_page()

