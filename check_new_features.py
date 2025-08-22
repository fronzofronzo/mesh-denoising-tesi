import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

# === CONFIG ===
folder_path = "new_samples/armadillo_0.2"  # <-- MODIFICA
features_to_check = ["alpha_n", "alpha_g", "area", "neighbors"]  # aggiungi altre se vuoi
output_prefix = "diagnostic_report"

# === FUNZIONI UTILI ===
def load_all_features(folder, features):
    data_dict = {f: [] for f in features}
    for file in os.listdir(folder):
        if file.endswith(".mat"):
            mat_path = os.path.join(folder, file)
            try:
                mat = loadmat(mat_path)
            except Exception as e:
                print(f"Errore caricando {file}: {e}")
                continue
            mat_features = mat["FEA"]
            data_dict["alpha_n"].append(mat_features[8,: ])
            data_dict["alpha_g"].append(mat_features[9,: ])
            data_dict["area"].append(mat_features[6,: ])
            data_dict["neighbors"].append(mat_features[7,: ])
    # Concatenazione di tutte le patch
    for k in data_dict:
        if len(data_dict[k]) > 0:
            data_dict[k] = np.concatenate(data_dict[k])
        else:
            data_dict[k] = np.array([])
    return data_dict

def feature_stats(arr):
    arr = arr[~np.isnan(arr)]  # rimuovi NaN
    return {
        "count": arr.size,
        "mean": np.mean(arr) if arr.size else np.nan,
        "std": np.std(arr) if arr.size else np.nan,
        "min": np.min(arr) if arr.size else np.nan,
        "max": np.max(arr) if arr.size else np.nan,
        "zeros": np.sum(arr == 0),
        "inf": np.sum(np.isinf(arr))
    }

# === MAIN ===
if __name__ == "__main__":
    data = load_all_features(folder_path, features_to_check)

    # Statistiche base
    stats_df = pd.DataFrame({feat: feature_stats(data[feat]) for feat in features_to_check})
    print("\n=== STATISTICHE FEATURE ===")
    print(stats_df)

    # Salva CSV
    stats_df.to_csv(f"{output_prefix}_stats.csv")

    # Istogrammi
    for feat in features_to_check:
        arr = data[feat]
        if arr.size > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(arr, bins=50, color="skyblue", edgecolor="black")
            plt.title(f"Istogramma {feat}")
            plt.xlabel(feat)
            plt.ylabel("Frequenza")
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_hist_{feat}.png", dpi=150)
            plt.close()

    # Heatmap correlazioni
    df = pd.DataFrame({feat: data[feat] for feat in features_to_check if data[feat].size > 0})
    if not df.empty:
        corr = df.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di correlazione")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_correlation_heatmap.png", dpi=150)
        plt.close()

    print(f"\n✅ Report generato: {output_prefix}_stats.csv + grafici PNG nella cartella corrente")
