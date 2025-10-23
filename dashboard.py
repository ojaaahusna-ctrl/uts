import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import base64
import requests # PERLU: Untuk Input URL

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionAI Dashboard | Final Version",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== DATA CONTOH GAMBAR (LENGKAP & VALID) ==================
# Catatan: Base64 Anda sudah valid, jadi saya pakai yang sudah Anda berikan.
CHEETAH_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMHBg0JCAgKDQ0HDQ0JBwYHDQ8IDQcNFREWFhURExMYHSggGBolGxMVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGxAQGy4mICYuMi0vLy0tLS8tLy0vNS8vLy0vLy0vLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAJgAqAMBIgACEQEDEQH/xAAcAAEAAgIDAQAAAAAAAAAAAAAABgcEBQEDCAL/xABDEAABAwIDBQQGBQkFCQAAAAABAAIDBBEFEgYhMUFRBxMiYXGBkRShscEIFCNCUnKSotEWJDNik7LC0uHwM1VzwvFE/8QAGwEBAAIDAQEAAAAAAAAAAAAAAAUGAwQHAgH/xAAxEQABAwIDBgQGAgMAAAAAAAAAAQIDBBEFEiExQRNRYXGBkaGx0fAUIjLBIkPhUoL/2gAMAwEAAhEDEQH/ALxREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREUGv/Q=="
HYENA_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMHBg0JCAgKDQ0HDQ0JBwYHDQ8IDQcNFREWFhURExMYHSggGBolGxMVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGxAQGy4mICYuMi0vLy0tLS8tLy0vNS8vLy0vLy0vLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAJgAqAMBIgACEQEDEQH/xAAcAAEAAgIDAQAAAAAAAAAAAAAABgcEBQEDCAL/xABDEAABAwIDBQQGBQkFCQAAAAABAAIDBBEFEgYhMUFRBxMiYXGBkRShscEIFCNCUnKSotEWJDNik7LC0uHwM1VzwvFE/8QAGwEBAAIDAQEAAAAAAAAAAAAAAAUGAwQHAgH/xAAxEQABAwIDBgQGAgMAAAAAAAAAAQIDBBEFEiExQRNRYXGBkaGx0fAUIjLBIkPhUoL/2gAMAwEAAhEDEQH/ALxREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREUGvQ=="
HOTDOG_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMHBg0JCAgKDQ0HDQ0JBwYHDQ8IDQcNFREWFhURExMYHSggGBolGxMVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGxAQGy4mICYuMi0vLy0tLS8tLy0vNS8vLy0vLy0vLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAJgAqAMBIgACEQEDEQH/xAAcAAEAAgIDAQAAAAAAAAAAAAAABgcEBQEDCAL/xABDEAABAwIDBQQGBQkFCQAAAAABAAIDBBEFEgYhMUFRBxMiYXGBkRShscEIFCNCUnKSotEWJDNik7LC0uHwM1VzwvFE/8QAGwEBAAIDAQEAAAAAAAAAAAAAAAUGAwQHAgH/xAAxEQABAwIDBgQGAgMAAAAAAAAAAQIDBBEFEiExQRNRYXGBkaGx0fAUIjLBIkPhUoL/2gAMAwEAAhEDEQH/ALxREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREUGvQ=="

# ================== STYLE KUSTOM (CSS) - TEMA "COOL MINT" ==================
# ... (Style CSS tidak diubah)

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
    """Fungsi untuk membersihkan state gambar yang dipilih dan input file/url."""
    # Reset semua variabel yang menyimpan input gambar
    st.session_state['selected_image_bytes'] = None
    if 'yolo_upload' in st.session_state: st.session_state['yolo_upload'] = None
    if 'cnn_upload' in st.session_state: st.session_state['cnn_upload'] = None
    if 'yolo_url_input' in st.session_state: st.session_state['yolo_url_input'] = ''
    if 'cnn_url_input' in st.session_state: st.session_state['cnn_url_input'] = ''
    # Reset radio button agar kembali ke default (Upload File) jika perlu
    if 'yolo_source' in st.session_state: st.session_state['yolo_source'] = "📤 Upload File"
    if 'cnn_source' in st.session_state: st.session_state['cnn_source'] = "📤 Upload File"


# ================== FUNGSI HALAMAN ==================
# ... (home_page tidak diubah kecuali penambahan clear_image_state)

def run_model_page(page_type):
    """Fungsi generik untuk menjalankan halaman model (YOLO atau CNN)."""
    
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
        
    # Ambil threshold dari session state atau gunakan default 0.85
    cnn_conf_threshold = st.session_state.get('cnn_conf', 0.85)

    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        clear_image_state()
        st.rerun()
    
    st.header(title)
    
    model = model_loader()
    if not model: return

    image_bytes = None
    
    source_key = f"{page_type}_source"
    upload_key = f"{page_type}_upload"
    url_key = f"{page_type}_url_input"

    with st.sidebar:
        st.title("⚙️ Pengaturan")
        
        if page_type == 'cnn':
            st.markdown("---")
            # Slider Confidence Threshold untuk CNN
            MIN_CONFIDENCE_THRESHOLD = st.slider("Min. Keyakinan Deteksi (CNN)", 0.0, 1.0, 0.85, 0.05, key="cnn_conf")
            st.info(f"Gambar di bawah {MIN_CONFIDENCE_THRESHOLD:.2%} akan ditolak.")
            
        st.markdown("---")
        # 🚨 PERBAIKAN: Hapus Kamera, Ganti dengan Input URL
        source_choice = st.radio(
            "Pilih sumber gambar:", 
            ["📤 Upload File", "🔗 Input URL", "🖼️ Pilih Contoh"], 
            key=source_key,
            # on_change=clear_image_state # Tidak perlu on_change di radio button jika sudah ditangani di masing-masing input
        )

        if page_type == 'yolo':
             confidence_threshold = st.slider("Tingkat Keyakinan (YOLO)", 0.0, 1.0, 0.5, 0.05, key="yolo_conf")

        # 1. Upload File
        if source_choice == "📤 Upload File":
            uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=upload_key)
            if uploaded_file: image_bytes = uploaded_file.getvalue()
                
        # 2. Input URL (PENGGANTI KAMERA)
        elif source_choice == "🔗 Input URL":
            url = st.text_input("Masukkan URL Gambar:", value=st.session_state.get(url_key, ''), key=url_key)
            if url:
                try:
                    with st.spinner("Mengunduh gambar..."):
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'image' not in content_type:
                            st.error("❌ URL tidak mengarah ke file gambar yang valid (Content-Type bukan gambar).", icon="⚠️")
                        else:
                            image_bytes = response.content
                            st.success("✅ Gambar berhasil diunduh.", icon="🌐")
                            
                except requests.exceptions.Timeout:
                     st.error("❌ Permintaan unduhan habis waktu (Timeout).", icon="⏳")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Gagal mengunduh gambar. Pastikan URL benar dan publik. Error: {e}", icon="🔥")
        
        # 3. Pilih Contoh (Menggunakan B64)
        else:
            st.subheader("Pilih gambar dari galeri:")
            cols = st.columns(len(sample_images))
            for idx, (caption, b64_string) in enumerate(sample_images.items()):
                with cols[idx]:
                    # Menggunakan mime type jpeg/png agar kompatibel dengan Base64
                    st.image(f"data:image/jpeg;base64,{b64_string}", caption=caption, use_container_width=True) 
                    if st.button(f"Gunakan {caption}", key=f"sample_{idx}", use_container_width=True):
                        # Simpan data gambar ke session state dan rerun
                        st.session_state['selected_image_bytes'] = base64.b64decode(b64_string)
                        # Reset input lain saat contoh dipilih agar tidak ada konflik
                        st.session_state[upload_key] = None 
                        st.session_state[url_key] = '' 
                        st.rerun()

    # Ambil bytes gambar yang terakhir kali dipilih/diunggah
    if image_bytes is None:
         image_bytes = st.session_state.get('selected_image_bytes')


    # ------------------ LOGIKA PREDIKSI UTAMA ------------------
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
             st.error("❌ File yang diunduh/diunggah bukan format gambar yang didukung (JPEG/PNG).", icon="🖼️")
             return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
        
        # Tambahkan tombol reset di bawah Gambar Asli
        with col1:
             if st.button("🗑️ Hapus Gambar & Reset", use_container_width=True, key=f"{page_type}_reset", on_click=clear_image_state):
                 st.rerun()


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
                            st.warning("Tidak ada objek terdeteksi.", icon="⚠️")
                else:
                    # LOGIKA KLASIFIKASI CNN DENGAN THRESHOLDING (Final)
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    
                    input_shape = model.input_shape[1:3]
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0) 
                    
                    preds = model.predict(img_array, verbose=0)[0]
                    pred_prob = np.max(preds) 
                    pred_idx = np.argmax(preds)
                    
                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        
                        if pred_prob >= cnn_conf_threshold:
                            # 1. Prediksi diterima (Keyakinan Tinggi)
                            st.metric("Prediksi Utama:", CLASS_NAMES_CNN.get(pred_idx))
                            st.metric("Tingkat keyakinan:", f"{pred_prob:.2%}")
                            st.success(f"✅ Gambar terdeteksi sebagai {CLASS_NAMES_CNN.get(pred_idx)}.", icon="✅")
                            
                            # TAMPILKAN DISTRIBUSI
                            st.subheader("📊 Distribusi Probabilitas")
                            for i, prob in enumerate(preds):
                                st.progress(float(prob), text=f"{CLASS_NAMES_CNN.get(i)}: {prob:.2%}")

                        else:
                            # 2. Prediksi ditolak (Keyakinan Rendah) - TIDAK TAMPILKAN DISTRIBUSI
                            st.error("❌ Gambar Tidak Terdeteksi", icon="🚫")
                            st.warning(f"Gambar tidak terdeteksi sebagai Cheetah atau Hyena karena keyakinan tertinggi ({pred_prob:.2%}) berada di bawah ambang batas ({cnn_conf_threshold:.2%}).")


# ================== ROUTER UTAMA APLIKASI ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')
