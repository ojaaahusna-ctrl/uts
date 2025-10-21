import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# ==================== MODEL PYTORCH YOLO ====================
if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model PyTorch - YOLO")
    
    try:
        # Load YOLO model
        @st.cache_resource
        def load_yolo():
            return YOLO('model/best.pt')
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        
        st.success("✅ Model YOLO berhasil dimuat!")
        
        # Info model di sidebar
        with st.sidebar.expander("📊 Info Model"):
            st.text(str(model.info()))
        
        # Upload gambar
        st.markdown("### Upload Gambar untuk Deteksi Objek")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='yolo'
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Deteksi Objek", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    # Prediksi
                    results = model(image)
                    
                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        result_img = results[0].plot()
                        st.image(result_img, use_column_width=True)
                    
                    # Detail deteksi
                    st.markdown("---")
                    st.subheader("📋 Detail Deteksi")
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        cols = st.columns(3)
                        for i, box in enumerate(boxes):
                            col_idx = i % 3
                            with cols[col_idx]:
                                st.metric(
                                    f"Objek {i+1}",
                                    model.names[int(box.cls)],
                                    f"{box.conf[0]:.1%}"
                                )
                    else:
                        st.info("ℹ Tidak ada objek terdeteksi")
                        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Pastikan file model ada di folder 'model/'")

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50")
    
    model = None
    
    try:
        # Load TensorFlow model
        @st.cache_resource
        def load_tensorflow():
            return tf.keras.models.load_model(
                'model/compressed.h5',
                compile=False
            )
        
        with st.spinner("Loading TensorFlow model..."):
            model = load_tensorflow()
        
        st.success("✅ Model berhasil dimuat!")
        
    except Exception as e:
        st.warning(f"⚠ Gagal load model: {str(e)[:150]}...")
        
        # Coba dengan safe_mode=False
        try:
            st.info("Mencoba metode alternatif...")
            
            @st.cache_resource
            def load_tensorflow_safe():
                import keras
                keras.config.disable_traceback_filtering()
                return tf.keras.models.load_model(
                    'model/Balqis Isaura_Laporan2.h5',
                    compile=False,
                    safe_mode=False
                )
            
            model = load_tensorflow_safe()
            st.success("✅ Model berhasil dimuat!")
            
        except Exception as e2:
            st.error(f"❌ Model tidak bisa dimuat")
            st.code(str(e2), language="text")
            st.info("""
            *Kemungkinan masalah:*
            - Versi TensorFlow/Keras tidak kompatibel
            - Model perlu disave ulang dengan format terbaru
            
            *Solusi sementara:* Gunakan model YOLO yang sudah berfungsi
            """)
    
    # Jika model berhasil di-load
    if model is not None:
        # Info model
        with st.sidebar.expander("📊 Architecture Model"):
            from io import StringIO
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            st.text(stream.getvalue())
        
        # Upload gambar
        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='tf'
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    # Preprocess
                    img_array = np.array(image.resize((224, 224)))
                    
                    # Convert RGBA to RGB jika perlu
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                    
                    # Prediksi
                    predictions = model.predict(img_array, verbose=0)
                    
                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]
                        
                        st.metric("Kelas Prediksi", f"Class {predicted_class}")
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Tampilkan semua probabilitas
                        with st.expander("📊 Lihat Semua Probabilitas"):
                            for i, prob in enumerate(predictions[0]):
                                st.progress(float(prob), text=f"Class {i}: {prob:.4f}")

# Footer
st.markdown("---")
st.markdown("📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")
