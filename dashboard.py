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
    page_title="VisionAI Dashboard | Girly Edition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== INITIALIZE SESSION STATE ==================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ================== DATA CONTOH GAMBAR ==================
# Base64 encoded images to avoid external URLs
CHEETAH_B64 = "iVBORw0KGgoAAAANSUhEUgAAARgAAAEYCAYAAACHjumMAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAACHxSURBVH... (data gambar sengaja dipotong agar ringkas)"
HYENA_B64 = "iVBORw0KGgoAAAANSUhEUgAAARgAAAEYCAYAAACHjumMAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAACHxSURBVH... (data gambar sengaja dipotong agar ringkas)"
HOTDOG_B64 = "iVBORw0KGgoAAAANSUhEUgAAARgAAAEYCAYAAACHjumMAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAACHxSURBVH... (data gambar sengaja dipotong agar ringkas)"

# ================== STYLE KUSTOM (CSS) - TEMA PINK ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Playfair+Display:wght@700&display=swap');

    /* Background utama dengan gradasi pink */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #ffdde1 0%, #ee9ca7 100%);
    }

    /* Kustomisasi header dengan animasi */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header {
        background-color: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        animation: fadeInDown 1s ease-out;
    }
    .header h1 {
        font-family: 'Playfair Display', serif;
        color: #581845;
        font-size: 3rem;
    }
    .header p {
        color: #900C3F;
        font-size: 1.2rem;
    }
    
    /* Kartu menu dengan efek hover */
    .menu-card {
        background-color: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.6);
        padding: 2rem 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease-in-out;
        height: 100%;
        backdrop-filter: blur(5px);
    }
    .menu-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 30px rgba(144, 12, 63, 0.15);
    }
    .menu-card h3 { color: #581845; font-family: 'Playfair Display', serif; }
    .menu-card p { color: #C70039; }

    /* Tombol utama */
    .stButton>button {
        background-color: #FF70AB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF5599;
    }

    /* Galeri contoh gambar */
    .sample-gallery img {
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.2s;
        border: 2px solid transparent;
    }
    .sample-gallery img:hover {
        transform: scale(1.05);
        border: 2px solid #FF70AB;
    }

</style>
""", unsafe_allow_html=True)

# ================== CACHE MODEL ==================
@st.cache_resource
def load_yolo_model():
    try: return YOLO("model/best.pt")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        return None

@st.cache_resource
def load_cnn_model():
    try: return tf.keras.models.load_model("model/compressed.h5", compile=True)
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== FUNGSI HALAMAN ==================

def home_page():
    """Menampilkan halaman menu utama."""
    st.markdown("""
    <div class="header">
        <h1>🌸 VisionAI Dashboard 🌸</h1>
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
    st.info("Proyek ini dibuat oleh **Balqis Isaura** sebagai bagian dari Ujian Tengah Semester.", icon="✨")

def run_model_page(page_type):
    """Fungsi generik untuk menjalankan halaman model (YOLO atau CNN)."""
    
    # Konfigurasi berdasarkan tipe halaman
    if page_type == 'yolo':
        title = "🌭 Deteksi Objek: Hotdog vs Not-Hotdog"
        model_loader = load_yolo_model
        sample_images = {"Contoh Hotdog": HOTDOG_B64}
        button_text = "🔍 Mulai Deteksi"
    else: # cnn
        title = "🐆 Klasifikasi Gambar: Cheetah vs Hyena"
        model_loader = load_cnn_model
        sample_images = {"Contoh Cheetah": CHEETAH_B64, "Contoh Hyena": HYENA_B64}
        button_text = "🔮 Lakukan Prediksi"

    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.header(title)
    
    model = model_loader()
    if not model: return

    image_bytes = None

    # === Sidebar dengan galeri contoh ===
    with st.sidebar:
        st.image("https://i.imgur.com/w08YV0a.png", width=100)
        st.title("⚙️ Pengaturan")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera", "🖼️ Pilih Contoh"], key=f"{page_type}_source")

        if page_type == 'yolo':
            st.markdown("---")
            confidence_threshold = st.slider("Tingkat Keyakinan", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")

    # Logika untuk mendapatkan gambar dari berbagai sumber
    if source_choice == "📤 Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"{page_type}_upload")
        if uploaded_file: image_bytes = uploaded_file.getvalue()
    elif source_choice == "📸 Ambil dari Kamera":
        camera_input = st.camera_input("Arahkan kamera", key=f"{page_type}_cam")
        if camera_input: image_bytes = camera_input.getvalue()
    else: # Pilih Contoh
        st.subheader("Pilih gambar dari galeri:")
        cols = st.columns(len(sample_images))
        for idx, (caption, b64_string) in enumerate(sample_images.items()):
            with cols[idx]:
                # Gunakan markdown untuk membuat gambar bisa diklik
                st.markdown(f'<div class="sample-gallery">', unsafe_allow_html=True)
                if st.button(caption, key=f"sample_{idx}"):
                    image_bytes = base64.b64decode(b64_string)
                st.image(f"data:image/jpeg;base64,{b64_string}", caption=caption, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True)
        
        placeholder = col2.empty()
        placeholder.info("Hasil akan muncul di sini setelah diproses.")

        if st.button(button_text, type="primary", use_container_width=True):
            with st.spinner("🧠 Menganalisis gambar..."):
                if page_type == 'yolo':
                    results = model(image, conf=confidence_threshold)
                    result_img_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                    with placeholder.container():
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img_rgb, use_column_width=True)
                    
                    st.subheader("📋 Detail Deteksi")
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes):
                            st.success(f"**Objek {i+1}:** `{model.names[int(box.cls)]}` | **Keyakinan:** `{box.conf[0]:.2%}`", icon="✅")
                    else:
                        st.warning("Tidak ada objek terdeteksi.", icon="⚠️")
                
                else: # cnn
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    input_shape = model.input_shape[1:3]
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0)
                    preds = model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(preds)
                    
                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Prediksi Utama:", CLASS_NAMES_CNN.get(pred_idx))
                        st.metric("Tingkat Keyakinan:", f"{preds[pred_idx]:.2%}")

                    st.subheader("📊 Distribusi Probabilitas")
                    for i, prob in enumerate(preds):
                        st.progress(float(prob), text=f"{CLASS_NAMES_CNN.get(i)}: {prob:.2%}")

# ================== ROUTER UTAMA APLIKASI ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')

