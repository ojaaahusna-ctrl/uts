import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Dashboard Deteksi & Klasifikasi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== STYLE KUSTOM (CSS) ==================
st.markdown("""
<style>
    /* Mengubah font utama */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    /* Memberi sedikit bayangan pada header */
    .stApp > header {
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Kustomisasi tombol utama */
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #4a4e69;
        font-weight: bold;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        border-color: #00aaff;
        box-shadow: 0 0 10px #00aaff;
    }
</style>
""", unsafe_allow_html=True)

# ================== HEADER UTAMA ==================
st.title("🎯 Dashboard Deteksi & Klasifikasi Gambar")
st.markdown("---")

# ================== CACHE MODEL ==================
@st.cache_resource
def load_yolo_model():
    """Memuat model YOLO dari file .pt dengan penanganan error."""
    try:
        model = YOLO("model/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: Pastikan file `model/best.pt` ada. Error: {e}", icon="🔥")
        return None

@st.cache_resource
def load_cnn_model():
    """Memuat model CNN dari file .h5 dengan penanganan error."""
    try:
        # Menggunakan compile=False untuk kompatibilitas, tapi compile=True lebih aman jika Anda tahu optimizer & loss-nya
        model = tf.keras.models.load_model("model/compressed.h5", compile=True)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: Pastikan file `model/compressed.h5` ada. Error: {e}", icon="🔥")
        return None

# ================== SIDEBAR ==================
with st.sidebar:
    st.image("https://i.imgur.com/G4f4bJb.png", width=100)
    st.title("🤖 Pengaturan Model")
    st.markdown("---")
    model_choice = st.radio(
        "Pilih Tugas:",
        ["🌭 Deteksi Hotdog (YOLO)", "🐆 Klasifikasi Cheetah/Hyena (CNN)"],
        captions=["Mendeteksi objek dalam gambar.", "Mengklasifikasikan isi gambar."]
    )
    st.markdown("---")
    st.info("Dibuat oleh **Balqis Isaura** untuk Proyek UTS.", icon="ℹ️")


# ===========================================================
# ================== BAGIAN YOLO ============================
# ===========================================================
if "YOLO" in model_choice:
    st.header("🧩 Deteksi Objek dengan YOLO")
    
    # === Kontrol di sidebar khusus untuk YOLO ===
    with st.sidebar:
        st.subheader("Pengaturan Deteksi")
        yolo_source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera"], key="yolo_source")
        confidence_threshold = st.slider("Tingkat Keyakinan (Confidence)", 0.0, 1.0, 0.5, 0.05)

    yolo_model = load_yolo_model()
    
    # Hanya lanjutkan jika model berhasil dimuat
    if yolo_model:
        image_bytes = None
        if yolo_source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"], key="yolo_upload")
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
        else:
            camera_input = st.camera_input("Arahkan kamera dan ambil gambar", key="yolo_cam")
            if camera_input:
                image_bytes = camera_input.getvalue()

        if image_bytes:
            # Buka gambar dari bytes menggunakan io
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Gambar Asli")
                st.image(image, use_column_width=True)

            if st.button("🔍 Jalankan Deteksi", type="primary", use_container_width=True):
                with st.spinner("🧠 Menganalisis gambar..."):
                    results = yolo_model(image, conf=confidence_threshold)
                    # Konversi hasil plot ke format RGB untuk ditampilkan di Streamlit
                    result_img_bgr = results[0].plot()
                    result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img_rgb, use_column_width=True)
                    
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
                            st.info("✅ Tidak ada objek terdeteksi dengan tingkat keyakinan yang dipilih.", icon="ℹ️")

# ===========================================================
# ================== BAGIAN CNN =============================
# ===========================================================
else:
    st.header("🧠 Klasifikasi Gambar dengan CNN")

    # === Kontrol di sidebar khusus untuk CNN ===
    with st.sidebar:
        st.subheader("Pengaturan Klasifikasi")
        cnn_source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera"], key="cnn_source")

    cnn_model = load_cnn_model()
    
    # Hanya lanjutkan jika model berhasil dimuat
    if cnn_model:
        # Definisikan nama kelas
        CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
        
        image_bytes = None
        if cnn_source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"], key="cnn_upload")
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
        else:
            camera_input = st.camera_input("Arahkan kamera dan ambil gambar", key="cnn_cam")
            if camera_input:
                image_bytes = camera_input.getvalue()

        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.subheader("🖼️ Gambar Asli")
                st.image(image, use_column_width=True)

            if st.button("🔮 Lakukan Prediksi", type="primary", use_container_width=True):
                with st.spinner("🧠 Memproses dan memprediksi..."):
                    # Ambil ukuran input dari model secara dinamis
                    input_shape = cnn_model.input_shape[1:3] # (height, width)

                    # Ubah ukuran, normalisasi, dan tambahkan dimensi batch
                    img_resized = image.resize(input_shape)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = cnn_model.predict(img_array, verbose=0)
                    confidence = float(np.max(predictions))
                    predicted_class_idx = int(np.argmax(predictions))
                    predicted_class_name = CLASS_NAMES_CNN.get(predicted_class_idx, f"Kelas {predicted_class_idx}")

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        # Gunakan st.metric untuk tampilan yang lebih baik
                        st.metric("Prediksi Utama", predicted_class_name)
                        st.metric("Tingkat Keyakinan", f"{confidence:.2%}")

                        # Beri feedback berdasarkan confidence
                        if confidence > 0.85:
                            st.success("Keyakinan Sangat Tinggi!", icon="✅")
                        elif confidence > 0.60:
                            st.warning("Keyakinan Cukup.", icon="⚠️")
                        else:
                            st.error("Keyakinan Rendah.", icon="❌")

                    # Tampilkan semua probabilitas dalam expander
                    with st.expander("📊 Lihat Semua Probabilitas"):
                        for i, prob in enumerate(predictions[0]):
                            class_name = CLASS_NAMES_CNN.get(i, f"Kelas {i}")
                            st.progress(float(prob), text=f"{class_name}: {prob:.2%}")

