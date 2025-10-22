import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# ====== Contoh Base64 Gambar Hotdog ======
HOTDOG_B64 = "PASTE_BASE64_STRING_GAMBAR_HOTDOG_KAMU_DI_SINI"

# ====== Fungsi Menjalankan Halaman Model ======
def run_model_page(model, page_type, class_names):
    st.title("🍔 Hotdog Classifier App")
    st.write("Unggah gambar, ambil dari kamera, atau gunakan contoh gambar yang tersedia!")

    # Input dari kamera
    camera_image = st.camera_input("📸 Ambil foto dengan kamera (opsional)")

    # Upload dari file
    uploaded_file = st.file_uploader("📁 Upload gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    # Gunakan contoh
    use_example = st.button("Gunakan Contoh Hotdog")

    image_bytes = None

    if use_example:
        try:
            image_bytes = base64.b64decode(HOTDOG_B64)
        except Exception as e:
            st.error(f"❌ Gagal memuat contoh gambar: {e}")

    elif uploaded_file is not None:
        image_bytes = uploaded_file.read()

    elif camera_image is not None:
        image_bytes = camera_image.getvalue()

    # Simpan di session_state agar tidak hilang saat tombol ditekan
    if image_bytes is not None:
        st.session_state[f"{page_type}_image_bytes"] = image_bytes

    # Ambil gambar dari session_state
    if f"{page_type}_image_bytes" in st.session_state:
        try:
            image = Image.open(io.BytesIO(st.session_state[f"{page_type}_image_bytes"])).convert("RGB")
            st.image(image, caption="📷 Gambar yang Diproses", use_column_width=True)

            # Preprocessing
            img = image.resize((224, 224))  # ubah sesuai ukuran input model
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Prediksi
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            st.success(f"### ✅ Hasil Prediksi: {predicted_class}")
            st.write(f"**Tingkat Kepercayaan:** {confidence:.2%}")

        except Exception as e:
            st.error(f"❌ Gagal membuka atau memproses gambar: {e}")
    else:
        st.info("Silakan ambil foto, upload gambar, atau gunakan contoh.")

# ====== Halaman Tentang ======
def about_page():
    st.title("ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan model deep learning berbasis **TensorFlow/Keras**
    untuk mengenali apakah suatu gambar berisi **Hotdog atau bukan Hotdog**.

    Anda bisa:
    - 📸 Mengambil gambar langsung dari kamera
    - 📁 Mengunggah gambar dari komputer
    - 🍔 Menggunakan contoh gambar Hotdog
    """)

# ====== Halaman Utama ======
def home_page():
    st.title("🎯 Selamat Datang di Hotdog Classifier!")
    st.write("""
    Aplikasi ini akan mendeteksi apakah gambar yang Anda masukkan merupakan **Hotdog** atau **Bukan Hotdog**.  
    Gunakan menu di sebelah kiri untuk mulai!
    """)

# ====== Main Program ======
def main():
    st.sidebar.title("📂 Navigasi")
    menu = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Model", "Tentang"])

    # Load model (pastikan model sudah dikonversi ke .keras)
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("model_hotdog.keras")

    model = load_model()
    class_names = ["Hotdog", "Not Hotdog"]

    if menu == "Beranda":
        home_page()
    elif menu == "Model":
        run_model_page(model, "hotdog_page", class_names)
    elif menu == "Tentang":
        about_page()

if __name__ == "__main__":
    main()
