import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  # OpenCV tetap dibutuhkan oleh ultralytics untuk beberapa operasi

# ================== KONFIGURASI HALAMAN ==================
# Menggunakan ikon dan layout yang lebih menarik
st.set_page_config(
    page_title="VisionAI Dashboard | Deteksi & Klasifikasi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================== STYLE KUSTOM (CSS) ==================
# CSS untuk memberikan tampilan modern pada berbagai elemen
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

    /* Kustomisasi tombol */
    .stButton>button {
        border-radius: 8px;
        border: 2px solid #4a4e69;
        background-color: transparent;
        color: #e0e0e0;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4a4e69;
        color: white;
        border-color: #9a8c98;
    }
    .stButton>button[kind="primary"] {
        background-color: #007bff;
        color: white;
        border: none;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #0056b3;
    }
    
    /* Kustomisasi container hasil */
    .results-container {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #161b22;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ================== CACHE MODEL ==================
# Menggunakan cache agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_yolo_model():
    """Memuat model YOLO dari file .pt."""
    try:
        model = YOLO("model/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        st.info("Pastikan file `best.pt` ada di dalam folder `model/`.")
        return None

@st.cache_resource
def load_cnn_model():
    """Memuat model CNN dari file .h5."""
    try:
        # Tidak gunakan compile=False agar lebih aman saat memuat model dengan optimizer state
        model = tf.keras.models.load_model("model/compressed.h5", compile=True)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        st.info("Pastikan file `compressed.h5` ada di dalam folder `model/`.")
        return None

# ================== SIDEBAR ==================
with st.sidebar:
    st.image("https://i.imgur.com/G4f4bJb.png", width=100) # Ganti dengan URL logo kamu jika ada
    st.title("⚙️ Pengaturan")
    st.markdown("---")
    
    # Pilihan model yang lebih deskriptif
    model_choice = st.radio(
        "Pilih Tugas:",
        ["🌭 Deteksi Hotdog vs Not-Hotdog (YOLO)", "🐆 Klasifikasi Cheetah vs Hyena (CNN)"],
        captions=["Model deteksi objek", "Model klasifikasi gambar"]
    )
    st.markdown("---")

    # Tambahkan input dari kamera, ini fitur "wah"
    st.header("Sumber Gambar")
    source_choice = st.radio("Pilih sumber:", ["📤 Upload File", "📸 Ambil dari Kamera"])
    
    # Slider confidence hanya untuk YOLO
    confidence_threshold = 0.5
    if "YOLO" in model_choice:
        st.markdown("---")
        confidence_threshold = st.slider("Tingkat Keyakinan (Confidence)", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    with st.expander("ℹ️ Tentang Proyek Ini"):
        st.write("""
        Dashboard ini adalah bagian dari Proyek Ujian Tengah Semester (UTS) Mata Kuliah Machine Learning.
        
        **Tujuan:**
        - **Model YOLO:** Mendeteksi objek 'hotdog' dan 'not-hotdog' dalam gambar.
        - **Model CNN:** Mengklasifikasikan gambar antara 'Cheetah' dan 'Hyena'.
        
        Dibuat oleh **Balqis Isaura** | Powered by Streamlit 🚀
        """)

# ================== HEADER UTAMA ==================
st.markdown("""
<div class="header">
    <h1>🤖 VisionAI Dashboard</h1>
    <p>Platform Interaktif untuk Deteksi dan Klasifikasi Gambar</p>
</div>
""", unsafe_allow_html=True)


# ================== Logika Utama Berdasarkan Pilihan Model ==================
if "YOLO" in model_choice:
    st.header("🌭 Deteksi Objek: Hotdog vs Not-Hotdog")
    yolo_model = load_yolo_model()
    
    if yolo_model:
        image_bytes = None
        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()

        else: # Ambil dari Kamera
            camera_input = st.camera_input("Arahkan kamera ke objek dan ambil gambar")
            if camera_input:
                image_bytes = camera_input.getvalue()

        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Gambar Asli")
                st.image(image, use_container_width=True, caption="Gambar yang di-upload")

            # Tombol untuk memulai deteksi
            if st.button("🔍 Mulai Deteksi", type="primary", use_container_width=True):
                with st.spinner("🧠 Menganalisis gambar..."):
                    results = yolo_model(image, conf=confidence_threshold)
                    result_img = results[0].plot() # .plot() menghasilkan gambar BGR (OpenCV)
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img_rgb, use_container_width=True, caption="Gambar dengan deteksi")
                
                # Menampilkan hasil dalam container yang lebih rapi
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
                        st.info("✅ Tidak ada objek yang terdeteksi sesuai ambang batas keyakinan.", icon="ℹ️")

else: # CNN SECTION
    st.header("🐆 Klasifikasi Gambar: Cheetah vs Hyena")
    cnn_model = load_cnn_model()

    # Definisikan nama kelas secara manual (HARUS SESUAI URUTAN SAAT TRAINING)
    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"} 

    if cnn_model:
        image_bytes = None
        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
        else:
            camera_input = st.camera_input("Arahkan kamera ke hewan dan ambil gambar")
            if camera_input:
                image_bytes = camera_input.getvalue()
        
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            col1, col2 = st.columns([0.6, 0.4]) # Kolom hasil lebih kecil
            with col1:
                st.subheader("🖼️ Gambar Asli")
                st.image(image, use_container_width=True, caption="Gambar yang akan diklasifikasi")

            # Tombol untuk memulai prediksi
            if st.button("🔮 Lakukan Prediksi", type="primary", use_container_width=True):
                with st.spinner("🧠 Memproses dan memprediksi..."):
                    input_shape = cnn_model.input_shape[1:3] # Ambil (height, width) misal (128, 128)
                    
                    # Preprocessing gambar
                    img_resized = image.resize(input_shape)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediksi
                    predictions = cnn_model.predict(img_array, verbose=0)
                    confidence = float(np.max(predictions))
                    predicted_class_idx = int(np.argmax(predictions))
                    predicted_class_name = CLASS_NAMES_CNN.get(predicted_class_idx, f"Kelas {predicted_class_idx} (Tidak Dikenal)")

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        
                        # Menggunakan container untuk hasil yang lebih menonjol
                        with st.container(border=True):
                            st.metric("Prediksi Utama", predicted_class_name)
                            st.metric("Tingkat Keyakinan", f"{confidence:.2%}")
                            
                            # Menambahkan status berdasarkan confidence
                            if confidence > 0.85:
                                st.success("Keyakinan Sangat Tinggi!", icon="✅")
                            elif confidence > 0.6:
                                st.warning("Keyakinan Cukup.", icon="⚠️")
                            else:
                                st.error("Keyakinan Rendah, hasil mungkin tidak akurat.", icon="❌")

                # Expander untuk probabilitas
                with st.expander("📊 Lihat Detail Probabilitas Semua Kelas"):
                    for i, prob in enumerate(predictions[0]):
                        class_name = CLASS_NAMES_CNN.get(i, f"Kelas {i}")
                        st.progress(float(prob), text=f"{class_name}: {prob:.2%}")
