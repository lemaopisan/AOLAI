import pickle
import pandas as pd
import numpy as np
import os

# ==========================================
# KONFIGURASI FILE
# ==========================================
# Pastikan nama file ini sesuai dengan yang Anda download dari Colab
MODEL_FILENAME = 'rf_malnutrition_model.pkl'

# ========================================== jerikho
# 1. FUNGSI LOAD MODEL
# ==========================================
def load_prediction_model(model_file=MODEL_FILENAME):
    """
    Memuat file model .pkl dari folder lokal.
    """
    # Cek apakah file ada di folder yang sama
    if not os.path.exists(model_file):
        return None
        
    try:
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# ==========================================dea
# 2. FUNGSI PREDIKSI UTAMA
# ==========================================
def predict_malnutrition(age, weight, height, muac, gender):
    """
    Menerima input data pasien, memprosesnya, dan mengembalikan hasil prediksi.
    
    Args:
        age (int): Umur dalam bulan
        weight (float): Berat dalam kg
        height (float): Tinggi dalam cm
        muac (float): LiLA (Lingkar Lengan Atas) dalam cm
        gender (str): 'Laki-laki' atau 'Perempuan'
        
    Returns:
        dict: Berisi 'status' (teks), 'code' (0/1), dan 'confidence' (persen)
    """
    
    # --- LANGKAH A: Load Model ---
    model = load_prediction_model()
    
    if model is None:
        return {
            "status": "Error: File model tidak ditemukan!",
            "prediction_code": -1,
            "confidence": 0
        }

    # --- LANGKAH B: Preprocessing Data ---
    # Kita harus samakan formatnya dengan saat training di Colab
    # Gender: Laki-laki = 1, Perempuan = 0
    gender_code = 1 if gender.lower() in ['laki-laki', 'male'] else 0
    
    # Buat DataFrame dengan nama kolom YANG SAMA PERSIS dengan training
    # Urutan kolom: Age, Weight, Height, MUAC, Gender_Code
    input_data = pd.DataFrame([[age, weight, height, muac, gender_code]], 
                              columns=['Age (months)', 'Weight_kg', 'Height_cm', 'MUAC_cm', 'Gender_Code'])
    
    # --- LANGKAH C: Prediksi ---
    try:
        # Prediksi Kelas (0 atau 1)
        prediction_class = model.predict(input_data)[0]
        
        # Prediksi Probabilitas (Seberapa yakin modelnya?)
        # model.predict_proba mengembalikan array seperti [0.05, 0.95]
        probs = model.predict_proba(input_data)[0]
        confidence_score = np.max(probs) * 100  # Ambil yang paling tinggi dan jadikan persen

        # --- LANGKAH D: Terjemahkan Hasil ---
        if prediction_class == 1:
            status_text = "⚠️ Terindikasi Gizi Buruk (Malnutrisi)"
            recommendation = "Segera konsultasikan ke Posyandu atau Dokter Anak."
        else:
            status_text = "✅ Status Gizi Normal"
            recommendation = "Pertahankan asupan gizi yang seimbang."
            
        return {
            "status": status_text,
            "prediction_code": int(prediction_class),
            "confidence": round(confidence_score, 2),
            "recommendation": recommendation
        }
        
    except Exception as e:
        return {
            "status": f"Error saat prediksi: {str(e)}",
            "prediction_code": -1,
            "confidence": 0
        }

# ========================================== jerikho
# 3. TEST AREA 
# ==========================================
if __name__ == "__main__":
    print("🧪 --- MULAI TEST MODEL.PY ---")
    
    # Kasus 1: Anak Sehat (Contoh)
    # Umur 24 bln, Berat 12kg (Bagus), LiLA 14cm (Bagus)
    print("\n1. Test Kasus Sehat:")
    result_sehat = predict_malnutrition(24, 12.0, 85.0, 14.0, "Laki-laki")
    print(result_sehat)
    
    # Kasus 2: Anak Malnutrisi (Contoh)
    # Umur 24 bln, Berat 8kg (Kurang), LiLA 11cm (Kecil)
    print("\n2. Test Kasus Malnutrisi:")
    result_sakit = predict_malnutrition(24, 8.0, 85.0, 11.0, "Perempuan")
    print(result_sakit)
    
    print("\n✅ Jika kedua hasil di atas muncul, model.py sudah siap!")