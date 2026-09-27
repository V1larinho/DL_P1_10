"""
evaluate_all.py - Gera results/metrics_val.csv de forma reprodutivel.

Avalia no conjunto de VALIDACAO:
  1. Aleatorio
  2. Maioria
  3. Tamanho (HistGBM, 3 features: width, height, ratio)
  4. Classico RGB sem perc (6 features: mean+std por canal, excluindo pretos)
  5. CNN (best_cnn.keras, limiar=0.20 determinado previamente no val)

Bootstrap por doente (n=1000, seed=42) para ICs de 95%.
Guarda: results/metrics_val.csv
"""

import os, random, warnings
warnings.filterwarnings("ignore")

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import torch
import keras
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# -- Reproducibilidade ---------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

VAL_THRESHOLD = 0.20   # fixo - determinado no val por F1, nunca reajustado
N_BOOT        = 1000

# -- Carregar dados ------------------------------------------------------------
print("A carregar NPZs ...")
data_train = np.load("data/processed/train.npz")
data_val   = np.load("data/processed/val.npz", allow_pickle=True)

X_train_raw = data_train['X']
X_val_raw   = data_val['X']
y_train     = data_train['y']
y_val       = data_val['y']
print(f"  Treino: {X_train_raw.shape}  Val: {X_val_raw.shape}")

# -- Metadata e grupos ---------------------------------------------------------
df_meta    = pd.read_csv("splits/metadata.csv")
df_train_s = pd.read_csv("splits/train.csv")
df_val_s   = pd.read_csv("splits/val.csv")

df_train_m = df_train_s.merge(df_meta, on=["relative_path", "class", "id_grupo"])
df_val_m   = df_val_s.merge(df_meta,   on=["relative_path", "class", "id_grupo"])

groups_val = df_val_m["id_grupo"].values
y_train_bl = (df_train_m["class"] == "Parasitized").astype(int).values
y_val_bl   = (df_val_m["class"]   == "Parasitized").astype(int).values

# Verificar alinhamento NPZ / CSV
paths_npz = [str(p) for p in data_val["paths"]]
paths_csv = list(df_val_m["relative_path"].values)
assert paths_npz == paths_csv, "ERRO: paths NPZ e CSV nao estao alinhados no val!"
print("  Alinhamento NPZ/CSV: OK")
assert (y_val == y_val_bl).all(), "ERRO: labels NPZ vs CSV divergem!"
print("  Labels consistentes: OK")

# -- Bootstrap helper ----------------------------------------------------------
def boot_metrics(y_true, y_prob, groups, threshold=0.5, n=N_BOOT):
    unique_g = np.unique(groups)
    grp_idx  = {g: np.where(groups == g)[0] for g in unique_g}
    point = {
        "Accuracy": accuracy_score(y_true, (y_prob >= threshold).astype(int)),
        "F1-Score": f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0),
        "ROC-AUC":  roc_auc_score(y_true, y_prob),
        "PR-AUC":   average_precision_score(y_true, y_prob),
    }
    boot = {k: [] for k in point}
    np.random.seed(SEED)
    for _ in range(n):
        bg = np.random.choice(unique_g, size=len(unique_g), replace=True)
        bi = np.concatenate([grp_idx[g] for g in bg])
        yt, yp = y_true[bi], y_prob[bi]
        if len(np.unique(yt)) < 2:
            continue
        yc = (yp >= threshold).astype(int)
        boot["Accuracy"].append(accuracy_score(yt, yc))
        boot["F1-Score"].append(f1_score(yt, yc, zero_division=0))
        boot["ROC-AUC"].append(roc_auc_score(yt, yp))
        boot["PR-AUC"].append(average_precision_score(yt, yp))
    fmt = {}
    for k in point:
        lo = np.percentile(boot[k], 2.5)
        hi = np.percentile(boot[k], 97.5)
        fmt[k] = f"{point[k]:.3f} ({lo:.3f}-{hi:.3f})"
    return fmt

# -- Baselines classicos -------------------------------------------------------
print("\nA extrair features classicas ...")

def get_classical_features(X_np):
    """6 features: mean+std RGB dos pixeis nao-pretos (sem perc)."""
    feats = []
    for img in X_np:
        mask = (img.sum(axis=2) > 0)
        if mask.sum() == 0:
            feats.append([0.0] * 6)
            continue
        vp     = img[mask]
        mean_c = vp.mean(axis=0) / 255.0
        std_c  = vp.std(axis=0) / 255.0
        feats.append(list(mean_c) + list(std_c))
    return np.array(feats)

X_train_class = get_classical_features(X_train_raw)
X_val_class   = get_classical_features(X_val_raw)

df_train_m["ratio"] = df_train_m["width"] / df_train_m["height"]
df_val_m["ratio"]   = df_val_m["width"]   / df_val_m["height"]
X_train_size = df_train_m[["width", "height", "ratio"]].values
X_val_size   = df_val_m[["width",   "height", "ratio"]].values

print("  A treinar LogReg (Classico RGB) ...")
lr_class = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train_class, y_train_bl)
preds_class = lr_class.predict_proba(X_val_class)[:, 1]

print("  A treinar HistGBM (Tamanho) ...")
hgbm = HistGradientBoostingClassifier(random_state=SEED).fit(X_train_size, y_train_bl)
preds_size = hgbm.predict_proba(X_val_size)[:, 1]

np.random.seed(SEED)
preds_rand = np.random.rand(len(y_val_bl))
preds_maj  = np.zeros(len(y_val_bl), dtype=float)

# -- CNN - carregar checkpoint e prever ---------------------------------------
print("\nA carregar CNN (best_cnn.keras) ...")
model = keras.models.load_model("checkpoints/best_cnn.keras")

X_val_f    = X_val_raw.astype(np.float32) / 255.0
X_train_f  = X_train_raw.astype(np.float32) / 255.0
mean_train = X_train_f.mean(axis=(0, 1, 2))
std_train  = X_train_f.std(axis=(0, 1, 2))
X_val_norm = (X_val_f - mean_train) / (std_train + 1e-7)

preds_cnn = model.predict(X_val_norm, batch_size=64, verbose=1).flatten()
print(f"  CNN preds: min={preds_cnn.min():.4f}  max={preds_cnn.max():.4f}")

# -- Avaliacao com bootstrap --------------------------------------------------
print(f"\nA fazer bootstrap (n={N_BOOT}) por doente ...")

models_val = [
    ("Aleatorio",               preds_rand,  y_val_bl, groups_val, 0.5),
    ("Maioria",                 preds_maj,   y_val_bl, groups_val, 0.5),
    ("Tamanho (HistGBM)",       preds_size,  y_val_bl, groups_val, 0.5),
    ("Classico RGB (sem area)", preds_class, y_val_bl, groups_val, 0.5),
    (f"CNN (thresh={VAL_THRESHOLD:.2f})", preds_cnn, y_val_bl, groups_val, VAL_THRESHOLD),
]

rows = []
for name, yp, yt, grps, thr in models_val:
    print(f"  {name} ...")
    r = boot_metrics(yt, yp, grps, threshold=thr)
    rows.append({"Model": name, **r})
    print(f"    AUC={r['ROC-AUC']}  F1={r['F1-Score']}  Acc={r['Accuracy']}")

df_out = pd.DataFrame(rows).set_index("Model")

# -- Guardar ------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
out_path = "results/metrics_val.csv"
df_out.to_csv(out_path)
print(f"\nGuardado: {out_path}")
print(df_out.to_string())

import datetime
ts = datetime.datetime.fromtimestamp(os.path.getmtime(out_path))
print(f"\nTimestamp metrics_val.csv: {ts}")
