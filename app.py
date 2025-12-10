# app.py - VERSI LENGKAP
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Growth Monitor AI", layout="wide")

# --- 1. SETUP & LOAD ---
if not os.path.exists('smart_growth_system.pkl'):
    st.error("Jalankan model.py dulu!")
    st.stop()

sys_data = joblib.load('smart_growth_system.pkl')
model_w = sys_data['model_weight']
model_h = sys_data['model_height']
standards = sys_data['standards']

DB_FILE = 'db_pertumbuhan.csv'

# --- 2. SISTEM PAKAR GIZI (Nutrition Rules) ---
def get_nutrition_advice(age, w_class, h_class):
    advice = {"status": "", "menu": [], "tips": []}
    
    # Base Advice by Age
    if age < 6:
        advice['menu'] = ["Hanya ASI Eksklusif."]
        advice['tips'] = ["Susui sesering mungkin.", "Ibu makan bergizi."]
    elif 6 <= age < 12:
        advice['menu'] = ["Bubur lumat (Nasi+Prohe+Lemak).", "Puree buah."]
        advice['tips'] = ["Tekstur bertahap (encer ke kental).", "Wajib Protein Hewani."]
    else:
        advice['menu'] = ["Makanan keluarga.", "Susu UHT/Pasteurisasi."]
        
    # Condition Advice (Weight)
    # 0,1: Kurus | 2: Normal | 3,4: Gemuk
    if w_class in [0, 1]:
        advice['status'] += "Berat Kurang. "
        advice['tips'].insert(0, "Tambahkan Minyak/Santan di makanan (Booster BB).")
        advice['menu'].append("Telur Puyuh/Ayam setiap hari.")
    elif w_class in [3, 4]:
        advice['status'] += "Berat Berlebih. "
        advice['tips'].insert(0, "Kurangi Gula (Susu kental manis, sirup).")
        advice['tips'].append("Perbanyak aktivitas fisik.")
        advice['menu'] = [m for m in advice['menu'] if 'Santan' not in m] # Hapus lemak
        
    # Condition Advice (Height)
    if h_class in [0, 1]: # Stunted
        advice['status'] += "Perawakan Pendek."
        advice['tips'].append("Wajib makanan tinggi Kalsium & Zinc.")
        advice['menu'].append("Ikan teri, Daging merah, Tahu/Tempe.")
        
    if not advice['status']: advice['status'] = "Gizi Baik & Normal"
    return advice

# --- 3. FUNGSI DATA ---
def save_data(nama, umur, berat, tinggi, stat_w, stat_h):
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
    else:
        df = pd.DataFrame(columns=['Tanggal','Nama','Umur','Berat','Tinggi','Status_Berat','Status_Tinggi'])
    
    new = pd.DataFrame([{
        'Tanggal': datetime.now().strftime("%Y-%m-%d"),
        'Nama': nama, 'Umur': umur, 'Berat': berat, 'Tinggi': tinggi,
        'Status_Berat': stat_w, 'Status_Tinggi': stat_h
    }])
    pd.concat([df, new], ignore_index=True).to_csv(DB_FILE, index=False)

def plot_chart(df, y_col, standard_df, gender_lbl, title):
    fig, ax = plt.subplots(figsize=(8,3))
    
    # Plot Standar WHO
    ax.plot(standard_df['Month'], standard_df['M'], color='green', alpha=0.3, label='Standar WHO')
    
    # Plot Data Anak
    ax.plot(df['Umur'], df[y_col], marker='o', color='blue', label='Anak')
    
    ax.set_title(title)
    ax.set_xlabel("Umur (Bulan)")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    return fig

# --- 4. UI ---
with st.sidebar:
    st.header("📝 Input Data")
    nama = st.text_input("Nama Anak")
    gender = st.selectbox("Gender", ["Male", "Female"])
    umur = st.number_input("Umur (bln)", 0, 60, 12)
    berat = st.number_input("Berat (kg)", 1.0, 50.0, 9.0)
    tinggi = st.number_input("Tinggi (cm)", 30.0, 120.0, 75.0)
    cek = st.button("Analisis")

st.title("🛡️ Sistem Pemantauan Tumbuh Kembang")

if cek and nama:
    # 1. Prediksi
    g_code = 1 if gender == "Male" else 0
    in_data = pd.DataFrame([[umur, berat, tinggi, g_code]], columns=['Age (months)','Weight_kg','Height_cm','Gender_Code'])
    
    res_w = model_w.predict(in_data)[0]
    res_h = model_h.predict(in_data)[0]
    
    # Mapping
    map_w = {0:"Sangat Kurus", 1:"Kurus", 2:"Normal", 3:"Gemuk", 4:"Obesitas"}
    map_h = {0:"Sangat Pendek", 1:"Pendek", 2:"Normal"}
    
    txt_w = map_w[res_w]
    txt_h = map_h[res_h]
    
    # 2. Rekomendasi
    rek = get_nutrition_advice(umur, res_w, res_h)
    
    # 3. Tampilan Atas
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hasil Analisis")
        st.info(f"Status Berat: {txt_w}")
        st.info(f"Status Tinggi: {txt_h}")
        save_data(nama, umur, berat, tinggi, txt_w, txt_h)
        
    with c2:
        st.subheader("💡 Rekomendasi Gizi")
        st.write(f"**Kesimpulan:** {rek['status']}")
        st.markdown("**Menu Saran:**")
        for m in rek['menu']: st.write(f"- {m}")
        st.markdown("**Tips:**")
        for t in rek['tips']: st.write(f"- {t}")

st.divider()

# --- 5. MONITORING HISTORY ---
st.subheader("📊 Riwayat Perkembangan")
if os.path.exists(DB_FILE):
    df_all = pd.read_csv(DB_FILE)
    df_anak = df_all[df_all['Nama'] == nama]
    
    if not df_anak.empty:
        # Tampilan Tabel Scrollable (Bukan cuma 5 baris)
        st.dataframe(df_anak, use_container_width=True)
        
        # Dual Grafik
        std = standards[gender]
        t1, t2 = st.tabs(["Grafik Berat", "Grafik Tinggi"])
        
        with t1:
            st.pyplot(plot_chart(df_anak, 'Berat', std['WFA'], gender, "Grafik Berat Badan"))
        with t2:
            st.pyplot(plot_chart(df_anak, 'Tinggi', std['HFA'], gender, "Grafik Tinggi Badan"))
            
    else:
        st.warning("Belum ada data untuk nama ini.")
else:
    st.info("Data kosong.")