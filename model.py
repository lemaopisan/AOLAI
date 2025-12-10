# model.py - FIXED VERSION
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

print("⚙️ Memulai Training Dual-Model System...")

# ==========================================
# 1. LOAD DATA WHO (DENGAN FIX SPASI)
# ==========================================
def load_who_standards():
    # Pastikan path sesuai. Jika file ada di folder 'who', gunakan prefix 'who/'
    # Jika file sejajar dengan model.py, hapus 'who/'
    base_path = "who/" if os.path.exists("who") else ""
    
    paths = {
        'Male': {
            'WFA': base_path + 'wfa_boys.csv', 
            'HFA_0_2': base_path + 'hfa_boys_0_2.csv', 
            'HFA_2_5': base_path + 'hfa_boys_2_5.csv'
        },
        'Female': {
            'WFA': base_path + 'wfa_girls.csv', 
            'HFA_0_2': base_path + 'hfa_girls_0_2.csv', 
            'HFA_2_5': base_path + 'hfa_girls_2_5.csv'
        }
    }
    
    standards = {}
    for gender, files in paths.items():
        # Load WFA
        try:
            wfa = pd.read_csv(files['WFA'], sep=';', decimal=',')
            wfa.columns = wfa.columns.str.strip() # Bersihkan spasi
        except FileNotFoundError:
            print(f"❌ File tidak ditemukan: {files['WFA']}")
            return None
        
        # Load HFA 0-2 & 2-5
        try:
            hfa1 = pd.read_csv(files['HFA_0_2'], sep=';', decimal=',')
            hfa1.columns = hfa1.columns.str.strip() # FIX: Bersihkan SEBELUM merge
            
            hfa2 = pd.read_csv(files['HFA_2_5'], sep=';', decimal=',')
            hfa2.columns = hfa2.columns.str.strip() # FIX: Bersihkan SEBELUM merge
            
            # Merge
            hfa = pd.concat([hfa1, hfa2], ignore_index=True)
            hfa = hfa.sort_values('Month').drop_duplicates(subset=['Month'])
        except FileNotFoundError:
             print(f"❌ File HFA tidak ditemukan untuk {gender}")
             return None
        
        standards[gender] = {'WFA': wfa, 'HFA': hfa}
    
    return standards

# ==========================================
# 2. FUNGSI LABELING (GROUND TRUTH)
# ==========================================
def get_z_score(val, params):
    # Pastikan params['L'], ['M'], ['S'] adalah float (scalar)
    L = float(params['L'])
    M = float(params['M'])
    S = float(params['S'])
    return ((val / M) ** L - 1) / (L * S)

def generate_labels(row):
    gender = row['Gender']
    age = int(row['Age (months)'])
    
    # Filter data tidak valid
    if age > 60: return pd.Series([None, None])
    
    std = standards.get(gender)
    if not std: return pd.Series([None, None])
    
    try:
        # Ambil parameter LMS
        wfa_p = std['WFA'].loc[std['WFA']['Month'] == age].iloc[0]
        hfa_p = std['HFA'].loc[std['HFA']['Month'] == age].iloc[0]
    except IndexError:
        return pd.Series([None, None])

    # Hitung Z-score
    z_weight = get_z_score(row['Weight_kg'], wfa_p)
    z_height = get_z_score(row['Height_cm'], hfa_p)
    
    # --- LOGIKA KELAS BERAT ---
    # 0=Sangat Kurus, 1=Kurus, 2=Normal, 3=Gemuk, 4=Obesitas
    if z_weight < -3: w_class = 0
    elif z_weight < -2: w_class = 1
    elif z_weight > 3: w_class = 4
    elif z_weight > 2: w_class = 3
    else: w_class = 2
    
    # --- LOGIKA KELAS TINGGI ---
    # 0=Sangat Pendek, 1=Pendek, 2=Normal
    if z_height < -3: h_class = 0
    elif z_height < -2: h_class = 1
    else: h_class = 2
    
    return pd.Series([w_class, h_class])

# ==========================================
# 3. EKSEKUSI TRAINING
# ==========================================
standards = load_who_standards()

if standards:
    print("✅ Standar WHO berhasil dimuat.")
    print("⏳ Sedang melatih model (Ini mungkin memakan waktu beberapa detik)...")
    
    try:
        # Load Dataset Ethiopia
        df = pd.read_csv('malnutrition_children_ethiopia.csv', sep=';')
        
        # Generate Label
        df[['Weight_Class', 'Height_Class']] = df.apply(generate_labels, axis=1)
        
        # Hapus data yang gagal dilabeli (Age > 60 atau error)
        df = df.dropna()
        
        # Persiapan Data Training
        df['Gender_Code'] = df['Gender'].map({'Male': 1, 'Female': 0})
        X = df[['Age (months)', 'Weight_kg', 'Height_cm', 'Gender_Code']]
        
        # Train Model 1: Berat
        rf_weight = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_weight.fit(X, df['Weight_Class'].astype(int))
        
        # Train Model 2: Tinggi
        rf_height = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_height.fit(X, df['Height_Class'].astype(int))
        
        # Simpan
        artifacts = {
            'model_weight': rf_weight,
            'model_height': rf_height,
            'standards': standards
        }
        joblib.dump(artifacts, 'smart_growth_system.pkl')
        print("💾 SUKSES! File 'smart_growth_system.pkl' berhasil dibuat.")
        print(f"   Akurasi Model Berat: {rf_weight.score(X, df['Weight_Class'].astype(int)):.2%}")
        
    except Exception as e:
        print(f"❌ Terjadi error saat training: {e}")
else:
    print("❌ Gagal memuat standar WHO. Periksa nama folder/file.")