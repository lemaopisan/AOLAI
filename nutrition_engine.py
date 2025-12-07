import os
import pandas as pd
import numpy as np

WHO_PATH = "who/"   # Folder WHO


# ==========================================================
# LOAD & NORMALISASI SEMUA FILE WHO
# ==========================================================
def load_who_tables():
    tables = {}

    for file in os.listdir(WHO_PATH):
        if not file.endswith(".csv"):
            continue

        key = file.replace(".csv", "")
        path = os.path.join(WHO_PATH, file)

        # Baca pakai semicolon (CSV dari Excel Indo)
        df = pd.read_csv(path, sep=";", dtype=str)

        # Normalisasi nama kolom
        df.columns = [c.strip().lower() for c in df.columns]

        # Rename ke format standar
        rename_map = {
            "month": "month",
            "age": "month",
            "age_mo": "month",
            "age (months)": "month",
            "l": "l",
            "m": "m",
            "s": "s",
        }
        df = df.rename(columns=rename_map)

        # Convert angka koma → titik
        for col in ["month", "l", "m", "s"]:
            if col in df.columns:
                df[col] = df[col].str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep hanya kolom penting
        keep = [c for c in ["month", "l", "m", "s"] if c in df.columns]
        df = df[keep]

        tables[key] = df

    return tables


WHO_TABLES = load_who_tables()


# ==========================================================
# PILIH FILE WHO SESUAI METRIK & USIA
# ==========================================================
def select_who_table(gender, metric, age_months):
    gender = gender.lower()

    if metric == "wfa":
        return f"wfa_{gender}"

    if metric == "bfa":
        return f"bfa_{gender}_0_2" if age_months <= 24 else f"bfa_{gender}_2_5"

    if metric == "hfa":
        return f"hfa_{gender}_0_2" if age_months <= 24 else f"hfa_{gender}_2_5"

    return None


# ==========================================================
# AMBIL L, M, S PALING DEKAT DENGAN USIA
# ==========================================================
def get_lms(table_name, age_months):
    df = WHO_TABLES[table_name].dropna(subset=["month"])

    idx = (df["month"] - age_months).abs().idxmin()
    row = df.loc[idx]

    return float(row["l"]), float(row["m"]), float(row["s"])


# ==========================================================
# HITUNG Z-SCORE
# ==========================================================
def calculate_zscore(measured, L, M, S):
    if L == 0:
        return np.log(measured / M) / S
    return ((measured / M) ** L - 1) / (L * S)


# ==========================================================
# KLASIFIKASI STATUS
# ==========================================================
def classify_z(z):
    if z < -3:
        return "Severely Malnourished"
    elif z < -2:
        return "Moderately Malnourished"
    elif z < -1:
        return "Mild Risk"
    else:
        return "Normal"


# ==========================================================
# REKOMENDASI KALORI SEDERHANA
# ==========================================================
def calorie_recommendation(age_months, weight, z_score):
    # --- Kebutuhan dasar menurut WHO energy requirement ---
    if age_months <= 3:
        kcal_per_kg = 115
    elif age_months <= 6:
        kcal_per_kg = 110
    elif age_months <= 12:
        kcal_per_kg = 100
    elif age_months <= 36:
        kcal_per_kg = 102
    elif age_months <= 60:
        kcal_per_kg = 94   # WHO range 90–95
    else:
        kcal_per_kg = 70

    base = kcal_per_kg * weight

    # --- Catch-up Growth (WHO Treatment of Malnutrition) ---
    if z_score < -3:       # Severe malnutrition
        return base * 1.40   # +40% energy
    elif z_score < -2:     # Moderate malnutrition
        return base * 1.20   # +20% energy
    else:
        return base          # Normal



# ==========================================================
# MAIN FUNCTION
# ==========================================================
def assess_child(name, gender, age_months, weight, height):

    # WFA Z-score
    wfa_table = select_who_table(gender, "wfa", age_months)
    L, M, S = get_lms(wfa_table, age_months)
    wfa_z = calculate_zscore(weight, L, M, S)
    wfa_status = classify_z(wfa_z)

    # HFA Z-score
    hfa_table = select_who_table(gender, "hfa", age_months)
    L, M, S = get_lms(hfa_table, age_months)
    hfa_z = calculate_zscore(height, L, M, S)
    hfa_status = classify_z(hfa_z)

    # BMI + BFA Z-score
    bmi = weight / (height/100)**2
    bfa_table = select_who_table(gender, "bfa", age_months)
    L, M, S = get_lms(bfa_table, age_months)
    bfa_z = calculate_zscore(bmi, L, M, S)
    bfa_status = classify_z(bfa_z)

    # --- gunakan WFA sebagai dasar pemberian kalori (lebih medis) ---
    calories = calorie_recommendation(age_months, weight, wfa_z)

    return {
        "name": name,
        "age_months": age_months,
        "weight": weight,
        "height": height,
        "BMI": bmi,
        "WFA_Z": wfa_z,
        "HFA_Z": hfa_z,
        "BFA_Z": bfa_z,
        "WFA_status": wfa_status,
        "HFA_status": hfa_status,
        "BFA_status": bfa_status,
        "Daily_Calorie_Need": calories
    }



# ==========================================================
# TEST
# ==========================================================
if __name__ == "__main__":
    child = assess_child(
        name="Adi",
        gender="boys",
        age_months=36,
        weight=20,
        height=88
    )
    print(child)
