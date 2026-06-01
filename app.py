import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
# Mengubah layout ke "wide" agar perbandingan 2 model lebih leluasa di layar
st.set_page_config(
    page_title="Klasifikasi Jajanan Tradisional Khas Bali",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
# Menambahkan sedikit gaya agar tombol dan kontainer terlihat lebih modern
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #6a1b9a;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }

div.stButton > button:first-child:hover {
    background-color: #98FF985;
    border-color: #98FF985;
    color: white;
}
    .model-title {
        text-align: center;
        color: #28a745;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'jajan_tradisional_bali.h5',
        compile=False
    )
    return model

model = load_model()

CLASS_NAMES = [
    'Cerorot',
    'Jajan Begina',
    'Jajan Matahari',
    'Jajan Non-Bai',
    'Jajan Reta',
    'Jajan Satuh',
    'Jajan Sirat',
    'Kaliadrem',
    'Laklak',
    'Lukis',
]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135703.png", width=100) 
    st.title("Tentang Aplikasi")
    st.info(
        "Aplikasi ini menggunakan teknologi *Deep Learning* untuk mendeteksi jenis jajanan tradisional khas Bali. "
        "Kami menggunakan arsitektur populer: *MobileNetV2*."
    )
    st.write("---")
    st.subheader("📌 Cara Penggunaan:")
    st.write("1. Upload foto jajanan tradisional khas Bali yang terlihat jelas.")
    st.write("2. Klik tombol *Mulai Deteksi*.")
    st.write("3. Lihat hasil klasifikasi.")

# --- HEADER UTAMA ---
st.markdown("<h1 style='text-align: center;'>Klasifikasi Jenis Jajanan Tradisional Khas Bali</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Sistem Klasifikasi jenis jajanan tradisional khas Bali</p>", unsafe_allow_html=True)
st.write("---")

# --- KONTEN UTAMA ---
# Bagi Layar
col_upload, col_result = st.columns([1, 2], gap="large")

with col_upload:
    st.subheader("📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Format yang didukung: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # Show gambar gambar 
        st.image(image, caption="Preview Gambar Jajanan", use_container_width=True)
        
        run_detection = st.button("Mulai Deteksi")
    else:
        run_detection = False
        st.warning("Silakan upload gambar jajan tradisional khas Bali terlebih dahulu.")

# --- Prediksi ---
with col_result:
    if uploaded_file and run_detection:
        st.subheader("📊 Hasil Analisis")

        # Preprocessing MobileNetV2
        img = image.resize((224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Jika saat training menggunakan MobileNetV2
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        with st.spinner("Memprediksi gambar..."):
            pred = model.predict(img)[0]

        idx = np.argmax(pred)
        confidence = pred[idx] * 100

        with st.container(border=True):
            st.metric(
                label="Hasil Klasifikasi",
                value=CLASS_NAMES[idx]
            )

            st.metric(
                label="Tingkat Keyakinan",
                value=f"{confidence:.2f}%"
            )

            with st.expander("Lihat Detail Probabilitas"):
                for i, name in enumerate(CLASS_NAMES):
                    st.write(f"{name}: {pred[i]*100:.2f}%")
                    st.progress(float(pred[i]))

    elif not uploaded_file:
        st.info("Hasil klasifikasi akan tampil di sini.")

       
