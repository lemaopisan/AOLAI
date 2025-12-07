# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(page_title="Growth Monitor AI", layout="wide")

# ==========================================
# 1. LOAD SYSTEM
# ==========================================
if not os.path.exists('malnutrition_system.pkl'):
    st.error("File sistem tidak ditemukan! Jalankan 'python model.py' terlebih dahulu.")
    st.stop()

# Load artifacts (Model + Standar WHO)
system_data = joblib.load('malnutrition_system.pkl')
model = system_data['model']
standards = system_data['standards']

DB_FILE = 'riwayat_kesehatan_anak.csv'

# ==========================================
# 2. FUNGSI PENDUKUNG
# ==========================================
def get_history():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=['Tanggal', 'Nama', 'Umur', 'Berat', 'Tinggi', 'Status'])

def save_data(nama, umur, berat, tinggi, status):
    df = get_history()
    new_row = pd.DataFrame({
        'Tanggal': [datetime.now().strftime("%Y-%m-%d")],
        'Nama': [nama], 'Umur': [umur], 
        'Berat': [berat], 'Tinggi': [tinggi], 
        'Status': [status]
    })
    pd.concat([df, new_row], ignore_index=True).to_csv(DB_FILE, index=False)

def plot_growth_curve(gender, age, weight, height, child_name):
    """Membuat grafik posisi anak dibanding standar WHO"""
    std_data = standards['Male' if gender == 'Laki-laki' else 'Female']['WFA']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot Kurva Standar (Median, -2SD, -3SD)
    ax.plot(std_data['Month'], std_data['M'], color='green', label='Median (Normal)', alpha=0.5)
    
    # Hitung garis batas bawah (-2 SD) secara manual mendekati visualisasi
    # (Penyederhanaan visual agar cepat dirender)
    lower_bound = std_data['M'] * 0.8 
    severe_bound = std_data['M'] * 0.7
    
    ax.fill_between(std_data['Month'], lower_bound, std_data['M'], color='yellow', alpha=0.1, label='Risiko Ringan')
    ax.fill_between(std_data['Month'], 0, lower_bound, color='red', alpha=0.1, label='Malnutrisi')
    
    # Plot Posisi Anak
    ax.scatter(age, weight, color='blue', s=100, zorder=5, label=f'Posisi {child_name}')
    ax.annotate(f"  {weight}kg", (age, weight))
    
    ax.set_title(f"Posisi Berat Badan {child_name} vs Standar WHO")
    ax.set_xlabel("Umur (Bulan)")
    ax.set_ylabel("Berat (kg)")
    ax.set_xlim(0, 60)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# ==========================================
# 3. USER INTERFACE (UI)
# ==========================================
st.title("🛡️ AI Child Growth Protector")
st.markdown("Sistem Deteksi Malnutrisi & Monitoring Pertumbuhan Anak")

# --- SIDEBAR INPUT ---
with st.sidebar:
    st.header("Data Anak")
    nama = st.text_input("Nama Anak")
    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur = st.slider("Umur (bulan)", 0, 60, 12)
    berat = st.number_input("Berat Badan (kg)", 1.0, 50.0, 8.0, step=0.1)
    tinggi = st.number_input("Tinggi Badan (cm)", 30.0, 120.0, 70.0, step=0.1)
    
    tombol_cek = st.button("🔍 Analisis Kesehatan")

# --- HALAMAN UTAMA ---
col1, col2 = st.columns([2, 1])

if tombol_cek and nama:
    # 1. Prediksi AI
    gender_code = 1 if gender == 'Laki-laki' else 0
    input_df = pd.DataFrame([[umur, berat, tinggi, gender_code]], 
                            columns=['Age (months)', 'Weight_kg', 'Height_cm', 'Gender_Code'])
    
    prediksi = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    
    # 2. Tampilan Hasil
    with col1:
        st.subheader("Hasil Diagnosis")
        if prediksi == 1:
            st.error(f"⚠️ PERINGATAN: Terdeteksi Risiko Malnutrisi (Confidence: {prob[1]:.1%})")
            st.markdown("""
            **Rekomendasi Tindakan:**
            - Tingkatkan asupan protein hewani (telur, ikan, daging).
            - Konsultasi ke Posyandu/Dokter terdekat.
            - Pantau berat badan 1 minggu lagi.
            """)
            status_text = "Malnutrisi"
        else:
            st.success(f"✅ KONDISI BAIK: Pertumbuhan Normal (Confidence: {prob[0]:.1%})")
            st.markdown("""
            **Saran:**
            - Pertahankan pola makan gizi seimbang.
            - Pastikan tidur cukup dan stimulasi bermain.
            """)
            status_text = "Normal"
            
        # Tampilkan Grafik Posisi Anak
        st.pyplot(plot_growth_curve(gender, umur, berat, tinggi, nama))
        
        # Simpan ke Database
        save_data(nama, umur, berat, tinggi, status_text)

    # 3. Riwayat Monitoring (Kanan)
    with col2:
        st.subheader("Riwayat Mingguan")
        df_log = get_history()
        if not df_log.empty:
            df_anak = df_log[df_log['Nama'] == nama]
            if not df_anak.empty:
                st.dataframe(df_anak[['Tanggal', 'Berat', 'Status']].tail(5), hide_index=True)
                
                # Sparkline chart kecil
                st.line_chart(df_anak.set_index('Tanggal')['Berat'])
            else:
                st.info("Data baru pertama kali.")
        else:
            st.info("Belum ada data historis.")

elif tombol_cek and not nama:
    st.warning("Mohon isi nama anak terlebih dahulu.")
else:
    st.info("👈 Masukkan data anak di sidebar untuk memulai analisis.")