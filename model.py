# model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Sedang memproses data... Mohon tunggu.")

# ==========================================
# 1. LOAD DATA STANDAR WHO
# ==========================================
def load_who_standards():
    # Pastikan file-file ini ada di satu folder yang sama
    files = {
        'who/bfa_boys_0_2': 'who/bfa_boys_0_2.csv', 'who/bfa_boys_2_5': 'who/bfa_boys_2_5.csv',
        'who/bfa_girls_0_2': 'who/bfa_girls_0_2.csv', 'who/bfa_girls_2_5': 'who/bfa_girls_2_5.csv',
        'who/hfa_boys_0_2': 'who/hfa_boys_0_2.csv', 'who/hfa_boys_2_5': 'who/hfa_boys_2_5.csv',
        'who/hfa_girls_0_2': 'who/hfa_girls_0_2.csv', 'who/hfa_girls_2_5': 'who/hfa_girls_2_5.csv',
        'who/wfa_boys': 'who/wfa_boys.csv', 'who/wfa_girls': 'who/wfa_girls.csv'
    }
    
    data = {}
    for key, path in files.items():
        try:
            df = pd.read_csv(path, sep=';', decimal=',')
            df.columns = df.columns.str.strip()
            data[key] = df
        except FileNotFoundError:
            print(f"Error: File {path} tidak ditemukan!")
            return None

    # Helper untuk menggabungkan 0-2 dan 2-5 tahun
    def merge_segments(df1, df2):
        return pd.concat([df1, df2], ignore_index=True).sort_values('Month').drop_duplicates(subset=['Month'])

    standards = {
        'Male': {
            'WFA': data['who/wfa_boys'],
            'HFA': merge_segments(data['who/hfa_boys_0_2'], data['who/hfa_boys_2_5']),
            'BFA': merge_segments(data['who/bfa_boys_0_2'], data['who/bfa_boys_2_5'])
        },
        'Female': {
            'WFA': data['who/wfa_girls'],
            'HFA': merge_segments(data['who/hfa_girls_0_2'], data['who/hfa_girls_2_5']),
            'BFA': merge_segments(data['who/bfa_girls_0_2'], data['who/bfa_girls_2_5'])
        }
    }
    return standards

standards = load_who_standards()

# ==========================================
# 2. PROSES DATA LATIH (ETHIOPIA) & LABELING
# ==========================================
# Kita butuh labeling (Ground Truth) berdasarkan Z-score
def calculate_label(row):
    gender = row['Gender']
    age = int(row['Age (months)'])
    if age > 60: return 'Exclude'
    
    tables = standards.get(gender)
    if not tables: return 'Exclude'
    
    # Ambil parameter LMS untuk Tinggi dan Berat
    try:
        wfa = tables['WFA'].loc[tables['WFA']['Month'] == age].iloc[0]
        hfa = tables['HFA'].loc[tables['HFA']['Month'] == age].iloc[0]
    except IndexError:
        return 'Exclude'

    # Rumus Z-score: ((X/M)^L - 1) / (L*S)
    def get_z(val, params):
        return ((val / params['M']) ** params['L'] - 1) / (params['L'] * params['S'])

    z_weight = get_z(row['Weight_kg'], wfa)
    z_height = get_z(row['Height_cm'], hfa)
    
    # Labeling: Malnutrisi jika Z < -2
    if z_weight < -2 or z_height < -2:
        return 1 # Malnutrisi
    else:
        return 0 # Normal

# Load Dataset Real
try:
    df_real = pd.read_csv('malnutrition_children_ethiopia.csv', sep=';')
    # Generate Target Variable (Status)
    df_real['Status'] = df_real.apply(calculate_label, axis=1)
    
    # Filter data bersih
    df_clean = df_real[df_real['Status'] != 'Exclude'].copy()
    df_clean['Status'] = df_clean['Status'].astype(int)
    
    print(f"Data siap: {len(df_clean)} baris.")
except FileNotFoundError:
    print("File dataset Ethiopia tidak ditemukan, pastikan nama file benar.")
    exit()

# ==========================================
# 3. TRAINING MODEL
# ==========================================
# Feature Engineering
df_clean['Gender_Code'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
X = df_clean[['Age (months)', 'Weight_kg', 'Height_cm', 'Gender_Code']]
y = df_clean['Status']

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X, y)

print("Model selesai dilatih.")

# ==========================================
# 4. SIMPAN MODEL & DATA REFERENCE
# ==========================================
# Kita simpan Model DAN Tabel Standar WHO agar app.py tidak perlu load CSV ulang
artifacts = {
    'model': rf_model,
    'standards': standards
}

joblib.dump(artifacts, 'malnutrition_system.pkl')
print("BERHASIL: Semua file disimpan ke 'malnutrition_system.pkl'")