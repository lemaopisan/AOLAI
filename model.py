# model.py - VERSI UPDATE (Dual Model & Multi-Class)
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("⚙️ Memulai Training Dual-Model System...")

# 1. LOAD DATA WHO
def load_who_standards():
    # Pastikan path folder 'who/' sesuai dengan struktur foldermu
    # Jika file ada di root folder, hapus prefix 'who/'
    paths = {
        'Male': {'WFA': 'who/wfa_boys.csv', 'HFA_0_2': 'who/hfa_boys_0_2.csv', 'HFA_2_5': 'who/hfa_boys_2_5.csv'},
        'Female': {'WFA': 'who/wfa_girls.csv', 'HFA_0_2': 'who/hfa_girls_0_2.csv', 'HFA_2_5': 'who/hfa_girls_2_5.csv'}
    }
    
    standards = {}
    for gender, files in paths.items():
        # Load WFA
        wfa = pd.read_csv(files['WFA'], sep=';', decimal=',')
        wfa.columns = wfa.columns.str.strip()
        
        # Load & Merge HFA
        hfa1 = pd.read_csv(files['HFA_0_2'], sep=';', decimal=',')
        hfa2 = pd.read_csv(files['HFA_2_5'], sep=';', decimal=',')
        hfa = pd.concat([hfa1, hfa2], ignore_index=True).sort_values('Month').drop_duplicates(subset=['Month'])
        hfa.columns = hfa.columns.str.strip()
        
        standards[gender] = {'WFA': wfa, 'HFA': hfa}
    
    return standards

try:
    standards = load_who_standards()
    print("✅ Standar WHO berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat WHO: {e}. Pastikan file ada di folder 'who/'.")
    exit()

# 2. FUNGSI LABELING (Membuat Kunci Jawaban)
def get_z_score(val, params):
    return ((val / params['M']) ** params['L'] - 1) / (params['L'] * params['S'])

def generate_labels(row):
    gender = row['Gender']
    age = int(row['Age (months)'])
    if age > 60: return pd.Series([None, None])
    
    std = standards.get(gender)
    if not std: return pd.Series([None, None])
    
    try:
        wfa_p = std['WFA'].loc[std['WFA']['Month'] == age].iloc[0]
        hfa_p = std['HFA'].loc[std['HFA']['Month'] == age].iloc[0]
    except:
        return pd.Series([None, None])

    z_weight = get_z_score(row['Weight_kg'], wfa_p)
    z_height = get_z_score(row['Height_cm'], hfa_p)
    
    # KELAS BERAT (Weight Class): 
    # 0=Sangat Kurus, 1=Kurus, 2=Normal, 3=Gemuk (Overweight), 4=Obesitas
    if z_weight < -3: w_c = 0
    elif z_weight < -2: w_c = 1
    elif z_weight > 3: w_c = 4
    elif z_weight > 2: w_c = 3
    else: w_c = 2
    
    # KELAS TINGGI (Height Class):
    # 0=Sangat Pendek, 1=Pendek, 2=Normal
    if z_height < -3: h_c = 0
    elif z_height < -2: h_c = 1
    else: h_c = 2
    
    return pd.Series([w_c, h_c])

# 3. TRAINING PROSES
print("⏳ Sedang melatih model...")
df = pd.read_csv('malnutrition_children_ethiopia.csv', sep=';') # Sesuaikan path jika perlu

# Generate Label
df[['Weight_Class', 'Height_Class']] = df.apply(generate_labels, axis=1)
df = df.dropna()

# Fitur Input
df['Gender_Code'] = df['Gender'].map({'Male': 1, 'Female': 0})
X = df[['Age (months)', 'Weight_kg', 'Height_cm', 'Gender_Code']]

# Model 1: Berat Badan (Multi-class)
rf_weight = RandomForestClassifier(n_estimators=100, random_state=42)
rf_weight.fit(X, df['Weight_Class'].astype(int))

# Model 2: Tinggi Badan
rf_height = RandomForestClassifier(n_estimators=100, random_state=42)
rf_height.fit(X, df['Height_Class'].astype(int))

# 4. SIMPAN
artifacts = {
    'model_weight': rf_weight,
    'model_height': rf_height,
    'standards': standards
}
joblib.dump(artifacts, 'smart_growth_system.pkl')
print("💾 File 'smart_growth_system.pkl' BERHASIL dibuat!")