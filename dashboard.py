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
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_image_bytes' not in st.session_state:
    st.session_state.selected_image_bytes = None

# ================== STYLE KUSTOM (CSS) - TEMA "COOL MINT" (FINAL) ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Playfair+Display:wght@700&display=swap');
    
    /* 1. LATAR BELAKANG CERAH (COOL MINT) - MEMAKSA MODE TERANG */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #E6FFFA 0%, #B2F5EA 100%);
        color: #2D3748; 
    }
    .stApp, .main, [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #E6FFFA 0%, #B2F5EA 100%); 
        color: #2D3748;
    }
    
    /* 2. KONSISTENSI WARNA TULISAN GELAP (Untuk Konten) */
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText, 
    [data-testid="stMarkdownContainer"],
    .stRadio > label,
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricValue"], 
    [data-testid="stAlert"]
    {
        color: #2D3748 !important; 
    }

    /* 3. PEMUSATAN KONTEN UTAMA (HOME PAGE) */
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
    
    /* 4. STYLE TOMBOL (MEMASTIKAN TEKS PUTIH) */
    .stButton>button {
        background-color: #319795;
        color: white !important; /* Teks tombol harus putih agar kontras */
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #2C7A7B; }
    
    /* 5. PEMUSATAN JUDUL HALAMAN */
    .main [data-testid="stHeader"] { text-align: center; }
    #home-container [data-testid="stSubheader"] { text-align: center; }
    
    
    /* 6. PERBAIKAN BACKGROUND KOMPONEN (YANG JADI HITAM) */
    
    /* Background Input Teks (URL) */
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        background-color: #FFFFFF !important;
        border-radius: 8px;
        border: 1px solid #B2F5EA;
        color: #2D3748 !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: #2D3748 !important;
    }

    /* Background File Uploader (Drag and Drop) */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #B2F5EA !important;
    }
    
    /* Teks di dalam File Uploader */
    [data-testid="stFileUploader"] section * {
        color: #2D3748 !important; 
    }
    
    /* Label di atas File Uploader */
    .stFileUploader > div > label {
        color: #2D3748 !important; 
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

# ================== FUNGSI UNTUK MENGHAPUS STATE GAMBAR ==================
def clear_image_state():
    """Fungsi untuk membersihkan state gambar yang dipilih."""
    st.session_state['selected_image_bytes'] = None
    
# --- GANTI FUNGSI reset_and_rerun LAMA ANDA DENGAN YANG INI ---
def reset_and_rerun():
    """Merupakan kombinasi reset state kustom dan pengaturan ulang state widget."""
    clear_image_state()
    
    # --- PERBAIKAN LOGIKA ERROR SAAT HAPUS (POIN 1 & 3) ---
    current_page = st.session_state.get('page', 'home') 
    
    if current_page == 'yolo' or current_page == 'cnn':
        source_key = f"{current_page}_source"
        url_key = f"{current_page}_url_input"
        upload_key = f"{current_page}_upload" # Kunci untuk file uploader

        # Periksa sumber yang sedang dipilih
        if source_key in st.session_state:
            selected_source = st.session_state[source_key]

            # 1. Jika sumbernya URL, reset text_input
            if selected_source == "🔗 Input URL Gambar":
                if url_key in st.session_state:
                    st.session_state[url_key] = ''
            
            # 2. Jika sumbernya Upload File, reset file_uploader (INI FIX POIN 1)
            elif selected_source == "📤 Upload File":
                if upload_key in st.session_state:
                    st.session_state[upload_key] = None # Mengosongkan file uploader
    
    # --- AKHIR PERBAIKAN ---
    st.rerun()
# --- AKHIR FUNGSI BARU ---


# ================== FUNGSI HALAMAN ==================

def home_page():
    """Menampilkan halaman menu utama."""
    
    # Bungkus seluruh konten di dalam st.container untuk pemusatan
    with st.container(border=False):
        # Menggunakan HTML/CSS untuk pemusatan
        st.markdown('<div id="home-container">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="header">
            <h1>✨ VisionAI Dashboard ✨</h1>
            <p>Platform Interaktif untuk Deteksi & Klasifikasi Gambar</p>
        </div>
        """, unsafe_allow_html=True)

        # Halaman subheader harus terpisah dari columns agar terpusat
        st.subheader("Pilih Tugas yang Ingin Dilakukan:", anchor=False)

        # Menggunakan columns untuk menu
        st.markdown('<div style="max-width: 800px;">', unsafe_allow_html=True) # Container pembatas lebar
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
        st.markdown('</div>', unsafe_allow_html=True) # Tutup container pembatas lebar

        st.markdown("---")
        
        # Info proyek (Nama sudah diganti)
        st.info("Proyek ini dibuat oleh **Raudhatul Husna** sebagai bagian dari Ujian Tengah Semester.", icon="🎓")
        
        st.markdown('</div>', unsafe_allow_html=True) # Tutup home-container

def run_model_page(page_type):
    """Fungsi generik untuk menjalankan halaman model (YOLO atau CNN)."""
    
    if page_type == 'yolo':
        title = "🌭 Deteksi Objek: Hotdog vs Not-Hotdog"
        model_loader = load_yolo_model
        button_text = "🔍 Mulai Deteksi"
    else:
        title = "🐆 Klasifikasi Gambar: Cheetah vs Hyena"
        model_loader = load_cnn_model
        button_text = "🔮 Lakukan Prediksi"
        
    cnn_conf_threshold = st.session_state.get('cnn_conf', 0.85)

    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        clear_image_state()
        st.rerun()
    
    st.header(title)
    
    # --- PENAMBAHAN INFO UNTUK CNN ---
    if page_type == 'cnn':
        st.info("💡 **Catatan:** Model ini hanya dilatih untuk mengenali **Cheetah** dan **Hyena**. Gambar di luar kategori ini akan ditolak jika keyakinannya di bawah ambang batas (atur di sidebar).", icon="💡")
    # --- AKHIR PENAMBAHAN ---
    
    # Klarifikasi Penggunaan Model (YOLO)
    if page_type == 'yolo':
         st.info("⚠️ Model deteksi ini hanya dilatih untuk mendeteksi **Hotdog**.", icon="💡")


    model = model_loader()
    if not model: return

    image_bytes = None
    
    source_key = f"{page_type}_source"
    upload_key = f"{page_type}_upload"
    url_key = f"{page_type}_url_input"
    cam_key = f"{page_type}_cam"

    with st.sidebar:
        st.title("⚙️ Pengaturan")
        
        if page_type == 'cnn':
            st.markdown("---")
            MIN_CONFIDENCE_THRESHOLD = st.slider("Min. Keyakinan Deteksi (CNN)", 0.0, 1.0, 0.85, 0.05, key="cnn_conf")
            
            # --- UBAH st.info MENJADI st.warning ---
            st.warning(f"""
            **Gunakan slider ini untuk 'menyaring' hasil.**
            
            * **Nilai TINGGI:** Lebih ketat. Bagus untuk menolak gambar non-Cheetah/Hyena, tapi berisiko menolak gambar asli yang buram/sulit.
            * **Nilai RENDAH:** Lebih longgar. Akan menerima lebih banyak gambar asli, tapi berisiko salah menerima gambar lain.
            
            Saat ini, prediksi di bawah **{MIN_CONFIDENCE_THRESHOLD:.2%}** akan ditolak.
            """, icon="⚖️")
            # --- AKHIR PERUBAHAN ---
            
        st.markdown("---")
        # Mengembalikan Kamera, Menghapus Pilih Contoh
        source_choice = st.radio(
            "Pilih sumber gambar:", 
            ["📤 Upload File", "📸 Ambil dari Kamera", "🔗 Input URL Gambar"], 
            key=source_key,
        )

        if page_type == 'yolo':
             confidence_threshold = st.slider("Tingkat Keyakinan (YOLO)", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")

        # 1. Upload File
        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=upload_key)
            if uploaded_file: 
                image_bytes = uploaded_file.getvalue()
                st.session_state['selected_image_bytes'] = image_bytes
                
        # 2. Ambil dari Kamera 
        elif source_choice == "📸 Ambil dari Kamera":
            camera_input = st.camera_input("Arahkan kamera", key=cam_key) 
            if camera_input: 
                image_bytes = camera_input.getvalue()
                st.session_state['selected_image_bytes'] = image_bytes
            st.info("⚠️ Fitur kamera mungkin tidak berfungsi jika aplikasi tidak berjalan di koneksi HTTPS.", icon="🛡️")

        # 3. Input URL Gambar
        elif source_choice == "🔗 Input URL Gambar":
            url = st.text_input("Masukkan URL Gambar:", value=st.session_state.get(url_key, ''), key=url_key)
            
            # KETERANGAN TAMBAHAN UNTUK USER (Sudah diringkas)
            st.info("""
            **Tips:** Gunakan **Direct Link** (berakhir `.jpg`, `.png`).
            
            **Cara Cepat:** Klik kanan pada gambar di web, lalu pilih **"Copy Image Address"** (atau "Salin Alamat Gambar").
            """, icon="💡")
            
            # Validasi URL web yang lengkap
            if url:
                if not re.match(r'https?://[^\s/$.?#].[^\s]*$', url):
                    st.error("❌ Masukkan URL web yang lengkap (diawali dengan http:// atau https://).", icon="⚠️")
                    image_bytes = None
                else:
                    try:
                        with st.spinner("Mengunduh gambar..."):
                            response = requests.get(url, timeout=10)
                            response.raise_for_status()
                            
                            content_type = response.headers.get('Content-Type', '').lower()
                            if 'image' not in content_type:
                                st.error("❌ URL tidak mengarah ke file gambar yang valid (Konten bukan gambar).", icon="⚠️")
                            else:
                                image_bytes = response.content
                                st.session_state['selected_image_bytes'] = image_bytes 
                                st.success("✅ Gambar berhasil diunduh.", icon="🌐")
                                
                    except requests.exceptions.Timeout:
                        st.error("❌ Permintaan unduhan habis waktu (Timeout).", icon="⏳")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Gagal mengunduh gambar. Pastikan URL benar, publik, dan merupakan Direct Link. Error: {e}", icon="🔥")


    # Ambil bytes gambar yang terakhir kali dipilih/diunggah/diambil
    if image_bytes is None:
         image_bytes = st.session_state.get('selected_image_bytes')


    # ------------------ LOGIKA PREDIKSI UTAMA ------------------
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
             st.error("❌ Data yang diunduh/diunggah bukan format gambar yang didukung (JPEG/PNG).", icon="🖼️")
             return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
            
            # Tombol Reset
            if st.button("🗑️ Hapus Gambar & Reset", use_container_width=True, key=f"{page_type}_reset"):
                 reset_and_rerun() 


        placeholder = col2.empty()
        placeholder.info("Tekan tombol di bawah untuk memproses gambar.")

        if st.button(button_text, type="primary", use_container_width=True, key=f"{page_type}_predict"):
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
                                st.success(f"**Objek {i+1}:** `{model.names[int(box.cls)]}` | **Keyakinan:** `{box.conf[0]:.2%}`", icon="✅")
                        else:
                            # --- PERBAIKAN POIN 2 (Not-Hotdog) ---
                            st.success("✅ **Hasil: Not-Hotdog.**\n\nModel tidak menemukan objek 'Hotdog' pada gambar ini.", icon="👍")
                            # --- AKHIR PERBAIKAN ---
                else:
                    # LOGIKA KLASIFIKASI CNN DENGAN THRESHOLDING
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    
                    input_shape = model.input_shape[1:3]
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0) 
                    
                    # === PERBAIKAN LOGIKA KLASIFIKASI (Sigmoid vs Softmax) ===
                    preds_output = model.predict(img_array, verbose=0)[0] 

                    if len(preds_output) == 1:
                        # KASUS 1: Model Sigmoid (Output Tunggal, e.g., [0.9])
                        pred_prob_raw = preds_output[0]
                        if pred_prob_raw > 0.5:
                            pred_idx = 1 # Hyena
                            pred_prob = pred_prob_raw
                        else:
                            pred_idx = 0 # Cheetah
                            pred_prob = 1.0 - pred_prob_raw 
                        
                        if pred_idx == 1:
                            preds_for_display = [1.0 - pred_prob, pred_prob] 
                        else:
                            preds_for_display = [pred_prob, 1.0 - pred_prob]
                    else:
                        # KASUS 2: Model Softmax (Output Ganda, e.g., [0.9, 0.1])
                        pred_prob = np.max(preds_output) 
                        pred_idx = np.argmax(preds_output)
                        preds_for_display = preds_output
                    # === AKHIR PERBAIKAN ===
                    
                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        
                        if pred_prob >= cnn_conf_threshold:
                            # 1. Prediksi diterima (Keyakinan Tinggi)
                            st.metric("Prediksi Utama:", CLASS_NAMES_CNN.get(pred_idx))
                            st.metric("Tingkat keyakinan:", f"{pred_prob:.2%}")
                            
                            # Emoji sudah 1
                            st.success(f"Gambar terdeteksi sebagai {CLASS_NAMES_CNN.get(pred_idx)}.", icon="✅")
                            
                            # TAMPILKAN DISTRIBUSI
                            st.subheader("📊 Distribusi Probabilitas")
                            for i, prob in enumerate(preds_for_display):
                                st.progress(float(prob), text=f"{CLASS_NAMES_CNN.get(i)}: {prob:.2%}")

                        else:
                            # 2. Prediksi ditolak (Keyakinan Rendah)
                            st.error("❌ Gambar Tidak Terdeteksi", icon="🚫")
                            st.warning(f"Gambar tidak terdeteksi karena keyakinan tertinggi ({pred_prob:.2%}) berada di bawah ambang batas ({cnn_conf_threshold:.2%}). Gambar mungkin bukan Cheetah atau Hyena.") 


# ================== ROUTER UTAMA APLIKASI ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')
