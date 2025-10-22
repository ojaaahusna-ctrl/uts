import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import base64
import os # Tambahkan import ini jika Anda perlu

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

# ================== DATA CONTOH GAMBAR (HARUS DIISI LENGKAP) ==================
# CATATAN PENTING: Ganti string placeholder di bawah ini dengan string Base64 LENGKAP
# dari file gambar Anda yang sebenarnya. Jika ini tidak diisi, gambar tidak akan muncul.
CHEETAH_B64 = "iVBORw0KGgoAAAANSUhEUgAA..." # [MASUKKAN B64 GAMBAR ASLI DI SINI]
HYENA_B64 = "iVBORw0KGgoAAAANSUhEUgAA..." # [MASUKKAN B64 GAMBAR ASLI DI SINI]
HOTDOG_B64 = "iVBORw0KGgoAAAANSUhEUgAA..." # [MASUKKAN B64 GAMBAR ASLI DI SINI]

# ================== STYLE KUSTOM (Tidak diubah) ==================
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

# ================== CACHE MODEL (Tidak diubah) ==================
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
        # Menambahkan parameter 'compile=False' jika terjadi error saat me-load model 
        # yang sudah disimpan dengan 'compile=True', tergantung versi TF.
        return tf.keras.models.load_model("model/compressed.h5", compile=True)
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== HALAMAN HOME (Tidak diubah) ==================
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
    st.info("Proyek ini dibuat oleh **Raudhatul Husna** sebagai bagian dari Ujian Tengah Semester.", icon="🎓")

# ================== HALAMAN MODEL (Perbaikan Utama di sini) ==================
def run_model_page(page_type):
    if page_type == 'yolo':
        title = "🌭 Deteksi Objek: Hotdog vs Not-Hotdog"
        model_loader = load_yolo_model
        # Perbaikan: Mengubah format string agar Streamlit tidak bingung
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
    
    # 🚨 PERBAIKAN KAMERA: Kamera Streamlit seringkali membutuhkan trik untuk bekerja 
    # di dalam loop/rerun. Kita akan coba memaksanya di sidebar.

    with st.sidebar:
        st.title("⚙️ Pengaturan")
        
        # Penambahan slider Confidence Threshold untuk CNN
        if page_type == 'cnn':
            st.markdown("---")
            # Ambang batas default 0.85
            MIN_CONFIDENCE_THRESHOLD = st.slider("Min. Keyakinan Deteksi (CNN)", 0.0, 1.0, 0.85, 0.05, key="cnn_conf")
            st.info(f"Gambar yang keyakinan prediksinya di bawah {MIN_CONFIDENCE_THRESHOLD:.2%} akan ditandai sebagai 'Tidak Terdeteksi'.")
            
        st.markdown("---")
        source_choice = st.radio("Pilih sumber gambar:", ["📤 Upload File", "📸 Ambil dari Kamera", "🖼️ Pilih Contoh"], key=f"{page_type}_source")

        if page_type == 'yolo':
             confidence_threshold = st.slider("Tingkat Keyakinan (YOLO)", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")


        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"{page_type}_upload")
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
        
        # 📸 Ambil dari Kamera: Kita gunakan kembali input kamera
        elif source_choice == "📸 Ambil dari Kamera":
            camera_input = st.camera_input("Arahkan kamera", key=f"{page_type}_cam")
            if camera_input:
                image_bytes = camera_input.getvalue()
        
        # 🖼️ Pilih Contoh: Menggunakan B64 yang telah diisi
        else:
            st.subheader("Pilih gambar dari galeri:")
            cols = st.columns(len(sample_images))
            for idx, (caption, b64_string) in enumerate(sample_images.items()):
                with cols[idx]:
                    # PERBAIKAN B64: Menggunakan f-string dengan data:image/jpeg;base64,
                    st.image(f"data:image/jpeg;base64,{b64_string}", caption=caption, use_container_width=True)
                    if st.button(f"Gunakan {caption}", key=f"sample_{idx}", use_container_width=True):
                        # Menggunakan base64.b64decode untuk mendapatkan bytes gambar
                        image_bytes = base64.b64decode(b64_string)
    
    # ------------------ LOGIKA PREDIKSI UTAMA ------------------
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
                    # LOGIKA DETEKSI YOLO (Tidak diubah, sudah baik)
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
                    # LOGIKA KLASIFIKASI CNN DENGAN THRESHOLDING (Perbaikan)
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    
                    # Ambil threshold dari sidebar
                    # Default: 0.85 jika tidak ada di session state (jika user tidak interaksi dengan slider)
                    cnn_conf_threshold = st.session_state.get('cnn_conf', 0.85) 
                    
                    input_shape = model.input_shape[1:3]
                    # Normalisasi: Penting! Jika model dilatih dengan /255, harus ada di sini.
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0) 
                    
                    preds = model.predict(img_array, verbose=0)[0]
                    pred_prob = np.max(preds) 
                    pred_idx = np.argmax(preds)
                    
                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        
                        # LOGIKA PENOLAKAN PREDIKSI (Thresholding)
                        if pred_prob >= cnn_conf_threshold:
                            # Prediksi terdeteksi
                            st.metric("Prediksi Utama:", CLASS_NAMES_CNN.get(pred_idx))
                            st.metric("Tingkat keyakinan:", f"{pred_prob:.2%}")
                            st.success(f"✅ Gambar terdeteksi sebagai {CLASS_NAMES_CNN.get(pred_idx)} dengan keyakinan yang tinggi.", icon="✅")
                        else:
                            # Prediksi gagal (di bawah ambang batas)
                            st.error("❌ Gambar Tidak Terdeteksi", icon="🚫")
                            st.warning(f"Keyakinan tertinggi ({pred_prob:.2%}) berada di bawah ambang batas deteksi ({cnn_conf_threshold:.2%}). Gambar mungkin bukan Cheetah atau Hyena.")
                        
                        st.subheader("📊 Distribusi Probabilitas")
                        for i, prob in enumerate(preds):
                            st.progress(float(prob), text=f"{CLASS_NAMES_CNN.get(i)}: {prob:.2%}")

# ================== ROUTER (Tidak diubah) ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')
