import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import re

# =============================
# Konfigurasi Halaman
# =============================
st.set_page_config(page_title="Klasifikasi Hewan - YOLO", layout="wide")

# =============================
# Inisialisasi Model
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # pastikan model kamu ada di folder yang sama

model = load_model()

# =============================
# Fungsi Utilitas
# =============================
def clear_image_state():
    for key in ["uploaded_file", "image_url", "result_img"]:
        if key in st.session_state:
            del st.session_state[key]

def go_home():
    st.session_state.page = "home"
    clear_image_state()
    st.rerun()

# =============================
# Fungsi Deteksi Gambar
# =============================
def detect_image(image):
    results = model(image)
    result_img = results[0].plot()

    # Cek channel (hindari error cvtColor)
    if result_img.shape[-1] == 3:
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    else:
        result_img_rgb = result_img

    # Ambil confidence tiap prediksi
    confs = []
    names = []
    for box in results[0].boxes:
        confs.append(float(box.conf[0]))
        cls_id = int(box.cls[0])
        names.append(results[0].names[cls_id])

    return result_img_rgb, confs, names

# =============================
# Tampilan Halaman
# =============================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.title("📸 Klasifikasi Hewan Menggunakan YOLO")
    st.markdown("Kelas: 1️⃣ Hyena | 2️⃣ Cheetah")
    st.divider()

    pilih = st.radio("Pilih sumber gambar:", ["Upload File", "Masukkan URL"], horizontal=True)

    if pilih == "Upload File":
        uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar diunggah", use_container_width=True)
            if st.button("🔍 Deteksi Gambar"):
                with st.spinner("Sedang memproses..."):
                    result_img, confs, names = detect_image(np.array(image))
                    st.session_state.result_img = result_img
                    st.session_state.confs = confs
                    st.session_state.names = names
                    st.session_state.page = "result"
                    st.rerun()

    elif pilih == "Masukkan URL":
        url = st.text_input("Masukkan URL gambar:")
        if url:
            if not re.match(r"^https?://[^\s]+$", url):
                st.warning("❌ URL tidak valid. Pastikan diawali dengan http:// atau https://")
            else:
                st.image(url, caption="Gambar dari URL", use_container_width=True)
                if st.button("🔍 Deteksi dari URL"):
                    with st.spinner("Sedang memproses..."):
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        img_data = np.asarray(bytearray(Image.open(url).tobytes()), dtype=np.uint8)
                        result_img, confs, names = detect_image(url)
                        st.session_state.result_img = result_img
                        st.session_state.confs = confs
                        st.session_state.names = names
                        st.session_state.page = "result"
                        st.rerun()

elif st.session_state.page == "result":
    st.title("📊 Hasil Deteksi")
    st.image(st.session_state.result_img, caption="Hasil Deteksi", use_container_width=True)

    st.subheader("Confidence per Prediksi")
    for n, c in zip(st.session_state.names, st.session_state.confs):
        st.write(f"- **{n}**: {c:.2f}")

    # Footer
    st.divider()
    st.markdown(
        "<center><small>Dibuat oleh <b>Raudhatul Husna</b> | Big Data</small></center>",
        unsafe_allow_html=True,
    )

    st.button("⬅️ Kembali ke Menu Utama", on_click=go_home)
