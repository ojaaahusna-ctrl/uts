import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# ================== KONFIGURASI DASHBOARD ==================
st.set_page_config(
    page_title="Dashboard YOLO + CNN",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Deteksi & Klasifikasi")
st.markdown("---")

# ================== PILIH MODEL ==================
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["🧩 PyTorch - YOLO", "🧠 TensorFlow - CNN (.h5)"]
)

# ===========================================================
# ================== YOLO SECTION ===========================
# ===========================================================
if model_choice == "🧩 PyTorch - YOLO":
    st.header("🧩 Model Deteksi Objek (YOLO)")
    
    try:
        @st.cache_resource
        def load_yolo_model():
            return YOLO("model/best.pt")

        with st.spinner("🔄 Memuat model YOLO..."):
            yolo_model = load_yolo_model()
        st.success("✅ Model YOLO berhasil dimuat!")

        uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"], key="yolo")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Gambar Asli")
                st.image(image, use_column_width=True)

            if st.button("🔍 Jalankan Deteksi", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    results = yolo_model(image)
                    result_img = results[0].plot()

                    with col2:
                        st.subheader("🎯 Hasil Deteksi YOLO")
                        st.image(result_img, use_column_width=True)

                    # Tampilkan daftar hasil
                    st.markdown("### 📋 Detail Deteksi")
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes):
                            st.write(f"**Objek {i+1}:** {yolo_model.names[int(box.cls)]} ({box.conf[0]:.2%})")
                    else:
                        st.info("Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Gagal memuat YOLO: {e}")
        st.info("Pastikan file `model/best.pt` tersedia di folder `model/`.")

# ===========================================================
# ================== CNN (H5) SECTION ========================
# ===========================================================
else:
    st.header("🧠 Model Klasifikasi (TensorFlow - CNN)")
    
    try:
        @st.cache_resource
        def load_cnn_model():
            # Tidak gunakan compile=True agar aman di TF versi apa pun
            model = tf.keras.models.load_model("model/compressed.h5", compile=False)
            return model
        
        with st.spinner("🔄 Memuat model CNN..."):
            cnn_model = load_cnn_model()
        st.success("✅ Model CNN berhasil dimuat!")

        st.write("📐 Input shape model:", cnn_model.input_shape)

        uploaded_file = st.file_uploader("📤 Upload gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="cnn")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Gambar Asli")
                st.image(image, use_column_width=True)

            if st.button("🔮 Prediksi Kelas", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    # Ambil ukuran input dari model (contoh: (None, 128, 128, 3))
                    input_shape = cnn_model.input_shape[1:3]

                    # Resize dan normalisasi
                    img_array = np.array(image.resize(input_shape)) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediksi
                    predictions = cnn_model.predict(img_array, verbose=0)
                    predicted_class = int(np.argmax(predictions))
                    confidence = float(np.max(predictions))

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", f"Class {predicted_class}")
                        st.metric("Confidence", f"{confidence:.2%}")

                    with st.expander("📊 Lihat Semua Probabilitas"):
                        for i, prob in enumerate(predictions[0]):
                            st.progress(float(prob), text=f"Class {i}: {prob:.4f}")

    except Exception as e:
        st.error("❌ Gagal memuat atau menjalankan model CNN.")
        st.code(str(e), language="text")
        st.info("""
        Coba periksa hal berikut:
        - File model `.h5` benar ada di folder `model/`
        - Versi TensorFlow kompatibel dengan format model
        - Model tidak kosong / corrupt
        """)

# ===========================================================
st.markdown("---")
st.markdown("📌 Dibuat oleh **Balqis Isaura** | Powered by Streamlit 🚀")
