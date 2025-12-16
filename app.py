import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ==========================================    udah terlalu kacau gainget pak ganti ganti mulu
# 0. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Growth Monitor AI + LiLA", layout="wide", page_icon="🛡️")

# ==========================================
# 1. SETUP & LOAD MODEL
# ==========================================
# A. Load Model Lama (Berat/Tinggi - WHO & Standards)
if not os.path.exists('smart_growth_system.pkl'):
    st.error("⚠️ File 'smart_growth_system.pkl' tidak ditemukan!")
    st.stop()

sys_data = joblib.load('smart_growth_system.pkl')
model_w = sys_data['model_weight']
model_h = sys_data['model_height']
standards = sys_data['standards']

# B. Load Model Baru (LiLA - Random Forest)
if not os.path.exists('rf_malnutrition_model.pkl'):
    st.error("⚠️ File 'rf_malnutrition_model.pkl' tidak ditemukan! Jalankan Phase 6 dulu.")
    st.stop()

with open('rf_malnutrition_model.pkl', 'rb') as file:
    rf_model_lila = pickle.load(file)

DB_FILE = 'db_pertumbuhan.csv'

# ==========================================
# 2. LOGIKA REKOMENDASI & STATUS
# ==========================================
def get_nutrition_advice(age, w_class, h_class, is_malnutrition):
    advice = {"status": "", "color": "", "tips": []}

    # PRIORITAS 1: GIZI BURUK (LiLA)
    if is_malnutrition == 1:
        advice['status'] = "GIZI BURUK / MALNUTRISI AKUT"
        advice['color'] = "error"
        advice['tips'] = [
            "🚨 **SEGERA KE DOKTER/PUSKESMAS.**",
            "Anak membutuhkan penanganan medis segera (F-75/F-100).",
            "Jangan paksa makan porsi besar, berikan sedikit tapi sering."
        ]
    
    # PRIORITAS 2: KURUS / PENDEK
    elif w_class in [0, 1] or h_class in [0, 1]:
        stat_list = []
        if w_class in [0, 1]: stat_list.append("Gizi Kurang (Kurus)")
        if h_class in [0, 1]: stat_list.append("Perawakan Pendek (Stunting)")
        
        advice['status'] = " + ".join(stat_list)
        advice['color'] = "warning"
        
        if w_class in [0, 1]:
            advice['tips'].append("👉 **Booster BB:** Tambahkan minyak/santan/margarin di setiap makanan.")
            advice['tips'].append("👉 **Double Protein:** Wajib Telur + Ikan/Ayam setiap makan.")
        if h_class in [0, 1]:
            advice['tips'].append("👉 **Kejar Tinggi:** Perbanyak Daging Merah & Susu (Zinc & Kalsium).")
            advice['tips'].append("👉 **Tidur:** Pastikan anak tidur lelap di malam hari (Hormon Pertumbuhan).")

    # PRIORITAS 3: GEMUK
    elif w_class in [3, 4]:
        advice['status'] = "RISIKO BERAT LEBIH (GEMUK)"
        advice['color'] = "info"
        advice['tips'] = [
            "👉 Kurangi gula (Susu kental manis, sirup, biskuit manis).",
            "👉 Ganti camilan dengan buah potong.",
            "👉 Ajak anak aktivitas fisik minimal 30 menit sehari."
        ]

    # PRIORITAS 4: NORMAL
    else:
        advice['status'] = "GIZI BAIK & NORMAL"
        advice['color'] = "success"
        advice['tips'] = [
            "✅ Pertahankan pola makan gizi seimbang.",
            "✅ Pantau terus pertumbuhan setiap bulan."
        ]
    
    return advice

# ==========================================
# 3. FUNGSI DATABASE
# ==========================================
def save_data(nama, umur, berat, tinggi, lila, stat_w, stat_h, stat_lila):
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        # Fix kolom jika file lama
        if 'LiLA' not in df.columns: df['LiLA'] = 0.0
        if 'Status_Gizi_AI' not in df.columns: df['Status_Gizi_AI'] = "-"
    else:
        df = pd.DataFrame(columns=['Tanggal','Nama','Umur','Berat','Tinggi','LiLA','Status_Berat','Status_Tinggi','Status_Gizi_AI'])
    
    tgl_skrg = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    new_data = pd.DataFrame([{
        'Tanggal': tgl_skrg,
        'Nama': nama, 'Umur': umur, 'Berat': berat, 'Tinggi': tinggi, 'LiLA': lila,
        'Status_Berat': stat_w, 'Status_Tinggi': stat_h, 'Status_Gizi_AI': stat_lila
    }])
    
    pd.concat([df, new_data], ignore_index=True).to_csv(DB_FILE, index=False)

# ==========================================
# 4. FUNGSI GRAFIK
# ==========================================

# A. Grafik Dual Axis (Riwayat Tanggal vs Berat/Tinggi)
def plot_dual_axis(df_anak):
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    df_anak['Tanggal'] = pd.to_datetime(df_anak['Tanggal'], format='mixed', dayfirst=False)
    df_sorted = df_anak.sort_values('Tanggal')
    dates = df_sorted['Tanggal']
    
    color1 = '#ff4b4b'
    ax1.set_ylabel('Berat (kg)', color=color1, fontweight='bold')
    ax1.plot(dates, df_sorted['Berat'], color=color1, marker='o', label='Berat')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.1)
    
    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    ax2.set_ylabel('Tinggi (cm)', color=color2, fontweight='bold')
    ax2.plot(dates, df_sorted['Tinggi'], color=color2, marker='s', linestyle='--', label='Tinggi')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig.autofmt_xdate()
    
    plt.title(f"Tren Pertumbuhan: {df_anak['Nama'].iloc[0]}")
    return fig

# B. Grafik LiLA (BARU!)
def plot_lila_chart(df_anak):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    
    df_anak['Tanggal'] = pd.to_datetime(df_anak['Tanggal'], format='mixed', dayfirst=False)
    df_sorted = df_anak.sort_values('Tanggal')
    
    # Plot Garis LiLA
    ax.plot(df_sorted['Tanggal'], df_sorted['LiLA'], marker='o', color='#d63384', linewidth=2, label='LiLA Anak')
    
    # Garis Batas Bahaya (12.5 cm)
    ax.axhline(y=12.5, color='yellow', linestyle='--', alpha=0.7, label='Batas Waspada (12.5cm)')
    ax.axhline(y=11.5, color='red', linestyle='--', alpha=0.7, label='Batas Bahaya (11.5cm)')
    
    # Area Merah (Arsir Bawah)
    ax.fill_between(df_sorted['Tanggal'], 0, 11.5, color='red', alpha=0.1)
    
    ax.set_ylabel("Lingkar Lengan (cm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig.autofmt_xdate()
    ax.legend(loc='upper left')
    ax.set_title("Tren Lingkar Lengan Atas (Indikator Gizi Buruk)")
    ax.grid(True, alpha=0.1)
    
    return fig

# C. Grafik Standar WHO
def plot_who_chart(df_anak, standard_df, y_col, title):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    max_age = df_anak['Umur'].max() + 5
    std_filtered = standard_df[standard_df['Month'] <= max_age]
    
    ax.plot(std_filtered['Month'], std_filtered['M'], color='#90ee90', linewidth=3, alpha=0.5, label='Standar WHO')
    ax.plot(df_anak['Umur'], df_anak[y_col], marker='o', color='blue', linewidth=2, label='Anak Anda')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Umur (Bulan)")
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return fig

# ==========================================
# 5. USER INTERFACE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📝 Input Data")
    nama = st.text_input("Nama Anak")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    
    umur = st.number_input("Umur (bln)", 0, 60, 12)
    berat = st.number_input("Berat (kg)", 1.0, 50.0, 9.0)
    tinggi = st.number_input("Tinggi (cm)", 30.0, 120.0, 75.0)
    
    st.markdown("---")
    st.write("💪 **Pengukuran LiLA**")
    muac = st.number_input("LiLA (cm)", 5.0, 30.0, 13.0, help="Indikator Malnutrisi Akut")
    
    cek = st.button("🔍 Analisis", use_container_width=True)

# ==========================================
# 6. HALAMAN UTAMA
# ==========================================
st.title("🛡️ Sistem Pemantauan Tumbuh Kembang")
st.markdown("---")

if cek and nama:
    g_code = 1 if gender == "Male" else 0
    
    # Prediksi
    in_old = pd.DataFrame([[umur, berat, tinggi, g_code]], columns=['Age (months)','Weight_kg','Height_cm','Gender_Code'])
    res_w = model_w.predict(in_old)[0]
    res_h = model_h.predict(in_old)[0]
    
    in_new = pd.DataFrame([[umur, berat, tinggi, muac, g_code]], columns=['Age (months)','Weight_kg','Height_cm','MUAC_cm','Gender_Code'])
    res_lila = rf_model_lila.predict(in_new)[0]
    
    # Mapping
    map_w = {0:"Sangat Kurus", 1:"Kurus", 2:"Normal", 3:"Gemuk", 4:"Obesitas"}
    map_h = {0:"Sangat Pendek", 1:"Pendek", 2:"Normal"}
    txt_w = map_w[res_w]
    txt_h = map_h[res_h]
    
    # Rekomendasi
    rek = get_nutrition_advice(umur, res_w, res_h, res_lila)
    save_data(nama, umur, berat, tinggi, muac, txt_w, txt_h, "Berisiko" if res_lila==1 else "Normal")
    
    # Output UI
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📊 Hasil Analisis")
        if rek['color'] == 'error': st.error(f"### {rek['status']}")
        elif rek['color'] == 'warning': st.warning(f"### {rek['status']}")
        elif rek['color'] == 'info': st.info(f"### {rek['status']}")
        else: st.success(f"### {rek['status']}")
        
        c_a, c_b = st.columns(2)
        with c_a: st.write(f"**Berat:** {txt_w}")
        with c_b: st.write(f"**Tinggi:** {txt_h}")
        st.write(f"**Status LiLA:** {'⚠️ Risiko Malnutrisi' if res_lila==1 else '✅ Aman/Normal'}")

    with col2:
        st.subheader("💡 Rekomendasi Dokter")
        for tip in rek['tips']: st.write(tip)

# ==========================================
# 7. GRAFIK RIWAYAT (TAB BARU)
# ==========================================
st.markdown("---")
st.subheader("📈 Grafik Riwayat")

if os.path.exists(DB_FILE):
    df_all = pd.read_csv(DB_FILE)
    if nama:
        df_anak = df_all[df_all['Nama'].str.lower() == nama.lower()]
        
        if not df_anak.empty:
            # ---> TAB 4 DITAMBAHKAN DI SINI <---
            t1, t2, t3, t4 = st.tabs(["📅 Tren (Dual Axis)", "⚖️ Berat (WHO)", "📏 Tinggi (WHO)", "💪 Grafik LiLA"])
            
            with t1:
                st.write("Tren Tanggal vs Berat/Tinggi")
                if len(df_anak) > 1: st.pyplot(plot_dual_axis(df_anak))
                else: st.info("Butuh min. 2 data tanggal berbeda.")
            
            with t2:
                std_now = standards[gender] 
                st.pyplot(plot_who_chart(df_anak, std_now['WFA'], 'Berat', "Berat Badan vs Standar WHO"))
                
            with t3:
                std_now = standards[gender]
                st.pyplot(plot_who_chart(df_anak, std_now['HFA'], 'Tinggi', "Tinggi Badan vs Standar WHO"))
                
            with t4:
                st.write("Tren Lingkar Lengan Atas (Deteksi Wasting)")
                if len(df_anak) > 1: st.pyplot(plot_lila_chart(df_anak))
                else: st.info("Butuh min. 2 data tanggal berbeda.")
                
            with st.expander("Lihat Data Tabel Lengkap"):
                st.dataframe(df_anak, use_container_width=True)
        else:
            st.warning(f"Belum ada data untuk anak bernama '{nama}'.")
    else:
        st.info("Masukkan nama anak di sidebar.")