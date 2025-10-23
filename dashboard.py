import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | YOLOv8 & CNN Intelligent Image Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== INISIALISASI SESSION STATE ==================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'cnn_conf' not in st.session_state:
    st.session_state.cnn_conf = 0.85
if 'need_rerun' not in st.session_state:
    st.session_state.need_rerun = False

# ================== CSS KHUSUS ==================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #eef2f3, #ffffff);
}
[data-testid="stSidebar"] {
    background-color: #f9fafb;
}
h1, h2, h3, h4 {
    color: #2C3E50;
    font-weight: 700;
}
.upload-text {
    color: #6b7280;
    font-size: 15px;
    margin-top: 8px;
}
.btn-primary {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
}
.btn-primary:hover {
    background-color: #1e40af !important;
}
</style>
""", unsafe_allow_html=True)

# ================== CACHE MODEL ==================
@st.cache_resource(show_spinner="🚀 Memuat model YOLOv8...")
def load_yolo():
    return YOLO("model/best.pt")

@st.cache_resource(show_spinner="🧠 Memuat model CNN...")
def load_cnn():
    return tf.keras.models.load_model("model/compressed.h5", compile=False)

# ================== FUNGSI ==================
def clear_image_state():
    keys = ['uploaded_image', 'camera_image', 'prediction', 'cnn_prediction', 'source_key']
    for k in keys:
        st.session_state.pop(k, None)
    st.session_state.need_rerun = True

# ================== HOME PAGE ==================
def home_page():
    st.title("👁️ VisionAI Dashboard")
    st.markdown("""
    ### YOLOv8 & CNN Intelligent Image Analyzer  
    Platform visual interaktif berbasis kecerdasan buatan untuk deteksi dan klasifikasi citra secara **real-time**.
    
    ---
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://cdn.dribbble.com/users/244516/screenshots/4860168/ai_camera.gif", use_container_width=True)
    with col2:
        st.markdown("""
        #### 🚀 Fitur Utama
        - **Object Detection (YOLOv8):** deteksi objek secara cepat dari kamera atau file.  
        - **Image Classification (CNN):** mengenali kategori gambar dengan akurasi tinggi.  
        - **Live Camera Mode:** uji deteksi secara langsung dari webcam.  
        - **Modern UI:** tampilan interaktif dan responsif untuk eksperimen AI visual.
        """)

    st.markdown("### 🔍 Pilih Mode Analisis:")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📸 Deteksi Objek (YOLOv8)", use_container_width=True, type="primary"):
            st.session_state.page = 'yolo'
            st.rerun()
    with col4:
        if st.button("🧩 Klasifikasi Gambar (CNN)", use_container_width=True, type="primary"):
            st.session_state.page = 'cnn'
            st.rerun()

# ================== YOLO PAGE ==================
def yolo_page():
    st.title("📸 Object Detection (YOLOv8)")
    yolo_model = load_yolo()

    st.markdown('<p class="upload-text">Unggah gambar atau gunakan kamera untuk mendeteksi objek.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("📂 Upload Gambar", type=["jpg", "jpeg", "png"], key="yolo_uploader")
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.source_key = "upload"

        st.divider()
        camera_photo = st.camera_input("📷 Ambil Gambar", key="yolo_camera")
        if camera_photo:
            st.session_state.camera_image = Image.open(camera_photo)
            st.session_state.source_key = "camera"

    with col2:
        st.subheader("⚙️ Pengaturan")
        conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, key="yolo_conf")
        st.info("💡 Gunakan nilai kecil (mis. 0.3) untuk deteksi lebih sensitif.")
        st.divider()
        if st.button("🔙 Kembali ke Beranda", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

    source_image = None
    if st.session_state.get("source_key") == "upload":
        source_image = st.session_state.uploaded_image
    elif st.session_state.get("source_key") == "camera":
        source_image = st.session_state.camera_image

    if source_image is not None:
        st.image(source_image, caption="Gambar Asli", use_container_width=True)
        img_array = np.array(source_image)
        results = yolo_model(img_array, conf=conf)

        plot_result = results[0].plot()
        if plot_result is not None:
            result_img_rgb = cv2.cvtColor(plot_result, cv2.COLOR_BGR2RGB)
            st.image(result_img_rgb, caption="Hasil Deteksi", use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Hapus Gambar", use_container_width=True):
                clear_image_state()
        with c2:
            st.download_button("💾 Simpan Hasil", data=cv2.imencode('.jpg', result_img_rgb)[1].tobytes(),
                               file_name="hasil_deteksi.jpg", mime="image/jpeg", use_container_width=True)

# ================== CNN PAGE ==================
def cnn_page():
    st.title("🧩 Image Classification (CNN)")
    cnn_model = load_cnn()

    st.markdown('<p class="upload-text">Unggah gambar untuk klasifikasi berbasis CNN.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📂 Upload Gambar", type=["jpg", "jpeg", "png"], key="cnn_uploader")
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.source_key = "upload"

    with col2:
        st.subheader("⚙️ Pengaturan")
        conf = st.slider("Confidence Threshold", 0.5, 1.0, st.session_state.cnn_conf, 0.01, key="cnn_conf_slider")
        st.session_state.cnn_conf = conf
        st.divider()
        if st.button("🔙 Kembali ke Beranda", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

    if "uploaded_image" in st.session_state:
        image = st.session_state.uploaded_image
        st.image(image, caption="Gambar Asli", use_container_width=True)

        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        prediction = cnn_model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"🧠 Prediksi: **Kelas {class_index}** | Akurasi: {confidence:.2%}")
        st.progress(float(confidence))

        st.divider()
        if st.button("🗑️ Hapus Gambar", use_container_width=True):
            clear_image_state()

# ================== NAVIGASI ==================
page = st.session_state.page
if page == 'home':
    home_page()
elif page == 'yolo':
    yolo_page()
elif page == 'cnn':
    cnn_page()

# ================== HANDLE RERUN ==================
if st.session_state.need_rerun:
    st.session_state.need_rerun = False
    st.rerun()
