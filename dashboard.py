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
# Variabel ini akan melacak halaman mana yang sedang aktif
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ================== STYLE KUSTOM (CSS) ==================
st.markdown("""
<style>
    /* Mengubah font utama */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Kustomisasi header */
    .header {
        background-color: #1a1a2e;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        border: 2px solid #4a4e69;
    }
    .header h1 {
        color: #e0e0e0;
        font-weight: 700;
    }
    .header p {
        color: #b0b0d0;
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
    }
    .menu-card:hover {
        border-color: #4a4e69;
        transform: translateY(-5px);
    }
    .menu-card h3 {
        color: #e0e0e0;
        margin-bottom: 1rem;
    }
    .menu-card p {
        color: #b0b0d0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ================== CACHE MODEL ==================
@st.cache_resource
def load_yolo_model():
    """Memuat model YOLO dari file .pt."""
    try:
        model = YOLO("model/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        return None

@st.cache_resource
def load_cnn_model():
    """Memuat model CNN dari file .h5."""
    try:
        model = tf.keras.models.load_model("model/compressed.h5", compile=True)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== FUNGSI UNTUK SETIAP HALAMAN ==================

def home_page():
    """Menampilkan halaman menu utama."""
    st.markdown("""
    <div class="header">
        <h1>🤖 VisionAI Dashboard</h1>
        <p>Platform Interaktif untuk Deteksi dan Klasifikasi Gambar</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Pilih Tugas yang Ingin Dilakukan:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="menu-card"><h3>🌭 Deteksi Objek</h3><p>Gunakan model YOLO untuk mendeteksi Hotdog vs Not-Hotdog dalam sebuah gambar.</p></div>', unsafe_allow_html=True)
            if st.button("Mulai Deteksi", use_container_width=True, key="yolo_nav"):
                st.session_state.page = 'yolo'
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown('<div class="menu-card"><h3>🐆 Klasifikasi Gambar</h3><p>Gunakan model CNN untuk mengklasifikasikan gambar antara Cheetah dan Hyena.</p></div>', unsafe_allow_html=True)
            if st.button("Mulai Klasifikasi", use_container_width=True, key="cnn_nav"):
                st.session_state.page = 'cnn'
                st.rerun()

    st.markdown("---")
    st.info("Proyek ini dibuat oleh **Balqis Isaura** sebagai bagian dari Ujian Tengah Semester.", icon="ℹ️")


def yolo_page():
    """Menampilkan halaman untuk deteksi YOLO."""
    # Tombol kembali
    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        st.rerun()

    st.header("🌭 Deteksi Objek: Hotdog vs Not-Hotdog")
    
    # === Sidebar untuk kontrol ===
    with st.sidebar:
        # Menggunakan LINK GAMBAR BARU yang berfungsi
        st.image("https://i.imgur.com/w08YV0a.png", width=100)
        st.title("⚙️ Pengaturan Deteksi")
        st.markdown("---")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera"], key="yolo_source")
        confidence_threshold = st.slider("Tingkat Keyakinan (Confidence)", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")
        st.markdown("---")
        with st.expander("ℹ️ Tentang Model Ini"):
            st.write("Model ini menggunakan arsitektur YOLOv8 untuk mendeteksi dua kelas: `hotdog` dan `not-hotdog`.")

    # === Logika Utama YOLO ===
    yolo_model = load_yolo_model()
    if not yolo_model: return

    image_bytes = None
    if source_choice == "📤 Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="yolo_upload")
        if uploaded_file: image_bytes = uploaded_file.getvalue()
    else:
        camera_input = st.camera_input("Arahkan kamera dan ambil gambar", key="yolo_cam")
        if camera_input: image_bytes = camera_input.getvalue()

    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True, caption="Gambar yang di-upload")

        # Tombol deteksi
        if st.button("🔍 Mulai Deteksi", type="primary", use_container_width=True):
            with st.spinner("🧠 Menganalisis gambar..."):
                results = yolo_model(image, conf=confidence_threshold)
                result_img = results[0].plot()
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                with col2:
                    st.subheader("🎯 Hasil Deteksi")
                    st.image(result_img_rgb, use_column_width=True, caption="Gambar dengan deteksi")

            # Detail hasil
            st.markdown("---")
            st.subheader("📋 Detail Deteksi")
            with st.container(border=True):
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        class_name = yolo_model.names[int(box.cls)]
                        confidence = box.conf[0]
                        st.write(f"**Objek {i+1}:** `{class_name}` dengan keyakinan **{confidence:.2%}**")
                else:
                    st.info("✅ Tidak ada objek terdeteksi.", icon="ℹ️")


def cnn_page():
    """Menampilkan halaman untuk klasifikasi CNN."""
    # Tombol kembali
    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        st.rerun()
        
    st.header("🐆 Klasifikasi Gambar: Cheetah vs Hyena")

    # === Sidebar untuk kontrol ===
    with st.sidebar:
        # Menggunakan LINK GAMBAR BARU yang berfungsi
        st.image("https://i.imgur.com/w08YV0a.png", width=100)
        st.title("⚙️ Pengaturan Klasifikasi")
        st.markdown("---")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera"], key="cnn_source")
        st.markdown("---")
        with st.expander("ℹ️ Tentang Model Ini"):
            st.write("Model ini menggunakan arsitektur Convolutional Neural Network (CNN) sederhana untuk mengklasifikasikan dua kelas hewan.")

    # === Logika Utama CNN ===
    cnn_model = load_cnn_model()
    if not cnn_model: return

    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}

    image_bytes = None
    if source_choice == "📤 Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="cnn_upload")
        if uploaded_file: image_bytes = uploaded_file.getvalue()
    else:
        camera_input = st.camera_input("Arahkan kamera dan ambil gambar", key="cnn_cam")
        if camera_input: image_bytes = camera_input.getvalue()

    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_column_width=True, caption="Gambar yang akan diklasifikasi")

        # Tombol prediksi
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

                with col2:
                    st.subheader("🎯 Hasil Prediksi")
                    with st.container(border=True):
                        st.metric("Prediksi Utama", predicted_class_name)
                        st.metric("Tingkat Keyakinan", f"{confidence:.2%}")
                        if confidence > 0.85: st.success("Keyakinan Sangat Tinggi!", icon="✅")
                        elif confidence > 0.6: st.warning("Keyakinan Cukup.", icon="⚠️")
                        else: st.error("Keyakinan Rendah.", icon="❌")

            with st.expander("📊 Lihat Detail Probabilitas"):
                for i, prob in enumerate(predictions[0]):
                    class_name = CLASS_NAMES_CNN.get(i, f"Kelas {i}")
                    st.progress(float(prob), text=f"{class_name}: {prob:.2%}")


# ================== ROUTER UTAMA APLIKASI ==================
# Memilih fungsi halaman mana yang akan dijalankan berdasarkan session state
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    yolo_page()
elif st.session_state.page == 'cnn':
    cnn_page()

