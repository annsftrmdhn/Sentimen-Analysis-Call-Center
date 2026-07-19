import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Set page configuration
st.set_page_config(
    page_title="MONEV Sentiment Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom premium styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        border-top: 5px solid #007bff;
    }
    .kpi-val {
        font-size: 32px;
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    .kpi-label {
        font-size: 14px;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Word lists for sentiment explanation
EXPLANATION_KEYWORDS = {
    'negatif': [
        'kecewa', 'kecewaan', 'mengecewakan', 'marah', 'kesal', 'jengkel',
        'lambat', 'lama', 'tidak jelas', 'bingung', 'susah', 'sulit',
        'error', 'gagal', 'reject', 'tolak', 'ditolak', 'hilang', 'rusak',
        'tidak ada', 'tidak bisa', 'gak bisa', 'pending', 'stuck',
        'tertahan', 'blokir', 'blok', 'scam', 'penipuan', 'hoax', 'fake',
        'bohong', 'buruk', 'kurang', 'lamban', 'lelet', 'rugi', 'salah'
    ],
    'positif': [
        'terima kasih', 'makasih', 'puas', 'membantu', 'bermanfaat',
        'cepat', 'ramah', 'baik', 'bagus', 'lancar', 'clear',
        'senang', 'suka', 'recommend', 'rekomendasi', 'thanks',
        'kereen', 'mantap', 'keren', 'love', 'hebat', 'mudah'
    ]
}

# Lazy load transformers pipeline inside the cached function to prevent Streamlit watcher import issues
@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier"
    )

try:
    classifier = load_sentiment_model()
except Exception as e:
    st.error(f"Gagal memuat model NLP: {e}. Silakan pastikan library 'torchvision' terinstal jika diperlukan (pip install torchvision).")
    classifier = None

# App Title (Professional tone, reduced emojis)
st.title("Dashboard Analisis Sentimen MONEV")
st.markdown("Aplikasi Dashboard Analisis Sentimen Tiket Pelanggan (Email, Live Chat, & Social Media) dengan Fitur Prediksi Real-time.")

# Sidebar - Navigation & File Upload
st.sidebar.header("Unggah Data & Navigasi")
uploaded_file = st.sidebar.file_uploader("Unggah File Hasil Sentimen (CSV)", type=["csv"])

# Load default dataset if no file uploaded
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File berhasil diunggah!")
else:
    # Try loading from modeling output directory
    df = load_data("output/sentiment_results.csv")
    if df is not None:
        st.sidebar.info("Menggunakan data default dari output/sentiment_results.csv")
    else:
        st.sidebar.warning("Data hasil sentimen belum tersedia. Silakan jalankan notebook modeling terlebih dahulu atau unggah file CSV.")

# Create tabs for different features (No emojis for a cleaner professional look)
tab1, tab2 = st.tabs(["Dashboard Analitik", "Prediksi Sentimen Baru"])

# Tab 1: Analytics Dashboard
with tab1:
    if df is not None:
        st.header("Metrik Utama (KPI)")
        
        # Calculate KPI values
        total_tickets = len(df)
        sentiment_counts = df['sentiment'].str.lower().value_counts()
        
        neg_pct = (sentiment_counts.get('negative', 0) / total_tickets) * 100
        pos_pct = (sentiment_counts.get('positive', 0) / total_tickets) * 100
        neu_pct = (sentiment_counts.get('neutral', 0) / total_tickets) * 100
        
        # Display KPI cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 5px solid #3498db;">
                    <div class="kpi-val">{total_tickets:,}</div>
                    <div class="kpi-label">Total Tiket</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 5px solid #e74c3c;">
                    <div class="kpi-val" style="color: #e74c3c;">{neg_pct:.1f}%</div>
                    <div class="kpi-label">Sentimen Negatif (Keluhan)</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 5px solid #2ecc71;">
                    <div class="kpi-val" style="color: #2ecc71;">{pos_pct:.1f}%</div>
                    <div class="kpi-label">Sentimen Positif</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 5px solid #95a5a6;">
                    <div class="kpi-val" style="color: #95a5a6;">{neu_pct:.1f}%</div>
                    <div class="kpi-label">Sentimen Netral</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Row 2: Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Distribusi Sentimen Keseluruhan")
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
            df['sentiment'].str.lower().value_counts().plot(
                kind='pie',
                autopct='%1.1f%%',
                colors=[colors.get(s, '#95a5a6') for s in df['sentiment'].str.lower().value_counts().index],
                startangle=90,
                ax=ax
            )
            ax.set_ylabel('')
            st.pyplot(fig)
            
        with col_chart2:
            st.subheader("Distribusi Sentimen per Channel Layanan")
            if 'channel' in df.columns:
                channel_col = 'channel'
            elif 'channel_source' in df.columns:
                channel_col = 'channel_source'
            else:
                channel_col = None
                
            if channel_col:
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.countplot(data=df, x=channel_col, hue='sentiment', palette=colors, ax=ax)
                ax.set_title("Sentimen berdasarkan Channel Layanan")
                ax.set_xlabel("Channel")
                ax.set_ylabel("Jumlah Tiket")
                st.pyplot(fig)
            else:
                st.info("Kolom channel tidak terdeteksi untuk menampilkan grafik channel.")

        st.markdown("---")
        st.subheader("Top Kategori Layanan dengan Persentase Sentimen Negatif Tertinggi")
        
        if 'category' in df.columns:
            ct = pd.crosstab(df['category'], df['sentiment'].str.lower(), normalize='index') * 100
            cat_counts = df['category'].value_counts()
            valid_cats = cat_counts[cat_counts >= 5].index
            ct_filtered = ct.loc[valid_cats]
            
            if 'negative' in ct_filtered.columns:
                top_neg_cats = ct_filtered.sort_values(by='negative', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                top_neg_cats['negative'].plot(kind='barh', color='#e74c3c', ax=ax)
                ax.set_xlabel("Persentase Sentimen Negatif (%)")
                ax.set_ylabel("Kategori")
                st.pyplot(fig)
            else:
                st.info("Data sentimen negatif tidak tersedia untuk ditampilkan per kategori.")
        else:
            st.info("Kolom kategori tidak ditemukan pada dataset.")

    else:
        st.info("Unggah dataset atau jalankan notebook untuk melihat visualisasi analitik.")

# Tab 2: Live Prediction Feature with Explanation
with tab2:
    st.header("Prediksi Sentimen Pertanyaan Pelanggan")
    st.markdown("Masukkan teks pertanyaan atau keluhan pelanggan untuk menganalisis sentimen dan mendapatkan rekomendasi penanganan.")
    
    user_input = st.text_area("Tulis atau Tempel Pertanyaan Pelanggan di Sini:", height=150, placeholder="Contoh: Saya kecewa sekali dengan pelayanan BEA CUKAI karena barang kiriman saya tertahan sangat lama...")
    
    if st.button("Prediksi Sentimen"):
        if not user_input.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        elif classifier is None:
            st.error("Model tidak dapat digunakan karena gagal dimuat. Pastikan seluruh dependensi terinstal.")
        else:
            with st.spinner("Menganalisis teks..."):
                # Run prediction
                pred = classifier(user_input[:512])[0]
                label = pred['label'].upper()
                score = pred['score']
                
                # Highlight keywords for explanation
                input_words = re.findall(r'\b\w+\b', user_input.lower())
                found_neg = [w for w in input_words if w in EXPLANATION_KEYWORDS['negatif']]
                found_pos = [w for w in input_words if w in EXPLANATION_KEYWORDS['positif']]
                
                # Visual output based on prediction
                if label == 'NEGATIVE':
                    st.error(f"**SENTIMEN NEGATIF** (Tingkat Keyakinan: {score*100:.2f}%)")
                    st.markdown("**Rekomendasi Aksi**: Teruskan ke Customer Service Specialist untuk penanganan prioritas.")
                    
                    # Explaining why
                    st.markdown("### Analisis Alasan Klasifikasi:")
                    if found_neg:
                        st.markdown(f"Model mendeteksi indikator kata bersentimen negatif kuat berikut pada input Anda: **{', '.join(set(found_neg))}**.")
                    else:
                        st.markdown("Model mengidentifikasi pola kalimat atau keluhan terstruktur yang mengindikasikan ketidakpuasan/permasalahan layanan, meskipun tidak menggunakan kata kunci keluhan secara eksplisit.")
                        
                elif label == 'POSITIVE':
                    st.success(f"**SENTIMEN POSITIF** (Tingkat Keyakinan: {score*100:.2f}%)")
                    st.markdown("**Rekomendasi Aksi**: Kirimkan ucapan terima kasih otomatis.")
                    
                    # Explaining why
                    st.markdown("### Analisis Alasan Klasifikasi:")
                    if found_pos:
                        st.markdown(f"Model mendeteksi indikator kata bersentimen positif kuat berikut pada input Anda: **{', '.join(set(found_pos))}**.")
                    else:
                        st.markdown("Model mendeteksi ungkapan apresiasi, kepuasan, atau konfirmasi penyelesaian masalah yang berjalan dengan baik.")
                        
                else:
                    st.info(f"**SENTIMEN NETRAL** (Tingkat Keyakinan: {score*100:.2f}%)")
                    st.markdown("**Rekomendasi Aksi**: Berikan jawaban template informatif standar.")
                    
                    # Explaining why
                    st.markdown("### Analisis Alasan Klasifikasi:")
                    st.markdown("Model mengklasifikasikan kalimat ini sebagai netral karena didominasi oleh pertanyaan prosedural, pencarian informasi umum, atau tidak mengandung kata/frasa bermuatan emosi positif maupun negatif yang signifikan.")
