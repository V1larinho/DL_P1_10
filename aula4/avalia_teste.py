import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# Garantir backend consistente
os.environ["KERAS_BACKEND"] = "torch"
import keras

TEST_CSV = "splits/test.csv"
VAL_CSV = "splits/val.csv"
VAL_X_CACHE = "aula3/cache/val_X_224.npy"
VAL_Y_CACHE = "aula3/cache/val_y.npy"

TEST_CACHE_DIR = "aula4/cache"
TEST_X_CACHE = os.path.join(TEST_CACHE_DIR, "test_X_224.npy")
TEST_Y_CACHE = os.path.join(TEST_CACHE_DIR, "test_y.npy")

RESULTS_DIR = "aula4/results"
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "test_predictions.csv")
ROC_CURVE_PNG = os.path.join(RESULTS_DIR, "roc_curve.png")
MODEL_PATH = "aula3/modelos/ft_melhor.keras"


def pad_and_resize(img_path, target_size=(224, 224)):
    """Exatamente o mesmo pre-processamento utilizado em train/val na aula 3."""
    with Image.open(img_path) as img:
        w, h = img.size
        max_dim = max(w, h)
        delta_w = max_dim - w
        delta_h = max_dim - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img_padded = ImageOps.expand(img, padding, fill=0)
        img_resized = img_padded.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img_resized, dtype=np.uint8)


def build_test_cache():
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    if os.path.exists(TEST_X_CACHE) and os.path.exists(TEST_Y_CACHE):
        print(f"Cache de teste ja existente encontrada em {TEST_CACHE_DIR}.")
        X_test = np.load(TEST_X_CACHE)
        y_test = np.load(TEST_Y_CACHE)
    else:
        print(f"A construir cache de teste a partir de {TEST_CSV}...")
        df = pd.read_csv(TEST_CSV)
        n = len(df)
        print(f"Total de amostras no CSV de teste: {n}")
        X_test = np.empty((n, 224, 224, 3), dtype=np.uint8)
        y_test = np.empty((n, 1), dtype=np.float32)

        t0 = time.time()
        for i, (_, row) in enumerate(df.iterrows()):
            img_path = os.path.join("data/raw", row["relative_path"])
            X_test[i] = pad_and_resize(img_path)
            # Classe 1 = Parasitized, Classe 0 = Uninfected
            y_test[i] = 1.0 if row["class"] == "Parasitized" else 0.0
            if (i + 1) % 1000 == 0 or (i + 1) == n:
                print(f"  Processadas {i + 1}/{n} imagens... ({time.time() - t0:.1f}s)")

        print("A guardar cache de teste no disco...")
        np.save(TEST_X_CACHE, X_test)
        np.save(TEST_Y_CACHE, y_test)
        print(f"Cache de teste guardada com sucesso! Tempo total: {time.time() - t0:.1f}s")

    # Confirmacoes obrigatorias
    df_test = pd.read_csv(TEST_CSV)
    assert len(X_test) == 4071, f"Esperado 4071 imagens, obtido {len(X_test)}"
    assert len(y_test) == 4071, f"Esperado 4071 rotulos, obtido {len(y_test)}"
    assert len(df_test) == 4071, f"Esperado 4071 linhas no CSV, obtido {len(df_test)}"
    
    # Confirmar correspondencia exata da ordem
    expected_y = np.array([1.0 if c == "Parasitized" else 0.0 for c in df_test["class"]]).reshape(-1, 1)
    np.testing.assert_array_equal(y_test, expected_y, err_msg="Ordem dos rotulos nao coincide com splits/test.csv!")
    print("Confirmacao concluida: 4071 imagens com ordem perfeitamente alinhada com splits/test.csv (SEM shuffle).\n")
    return X_test, y_test, df_test


def compute_val_threshold_table(model):
    print("=" * 70)
    print("ANALISE DE LIMIARES NO CONJUNTO DE VALIDACAO (VAL)")
    print("=" * 70)
    if not (os.path.exists(VAL_X_CACHE) and os.path.exists(VAL_Y_CACHE)):
        raise FileNotFoundError(f"Cache de validacao nao encontrada em {VAL_X_CACHE}")

    X_val = np.load(VAL_X_CACHE)
    y_val = np.load(VAL_Y_CACHE).flatten()

    print(f"A inferir conjunto de validacao ({len(X_val)} amostras)...")
    val_probs = model.predict(X_val, batch_size=64, verbose=0).flatten()

    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
    rows = []

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall / Sensibilidade
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # 1 - Especificidade
        precision = tp / (tp + fn) if (tp + fp) > 0 else 0.0
        # Precision correta: TP / (TP + FP)
        precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        youden = tpr - fpr

        rows.append({
            "Limiar": t,
            "TPR (Sensibilidade/Recall)": tpr,
            "FPR (1 - Especificidade)": fpr,
            "Precision (Parasitized)": precision_real,
            "TPR - FPR (Youden J)": youden,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    df_thresh = pd.DataFrame(rows)
    print("\nTabela de Limiares no VAL:")
    print(df_thresh[["Limiar", "TPR (Sensibilidade/Recall)", "FPR (1 - Especificidade)", "Precision (Parasitized)", "TPR - FPR (Youden J)"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 70)
    print("Nota clinica para rastreio: Maximizar TPR (minimizar FN / doentes com malaria falhados).")
    return df_thresh


def run_test_inference_and_metrics(model, X_test, y_test, df_test):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Verificacao do enunciado: roc_auc_score mock
    mock_true = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    mock_prob = [0.95, 0.85, 0.78, 0.70, 0.62, 0.55, 0.45, 0.35, 0.20, 0.10]
    mock_score = roc_auc_score(mock_true, mock_prob)
    print("=" * 70)
    print("VERIFICACAO DO ENUNCIADO (ROC-AUC MOCK)")
    print(f"roc_auc_score do exemplo de teste: {mock_score:.4f} (Esperado: 0.84)")
    assert abs(mock_score - 0.84) < 1e-5, f"Erro: mock auc score deu {mock_score}, esperado 0.84"
    print("Verificacao do enunciado bem sucedida!")
    print("=" * 70)

    # UMA predicao no teste
    print("\nA executar UMA predicao no conjunto de teste com o modelo final...")
    t0 = time.time()
    test_probs = model.predict(X_test, batch_size=64, verbose=1).flatten()
    print(f"Predicao concluida em {time.time() - t0:.2f}s!")

    # Guardar logo y_true, y_prob e identificadores no CSV
    df_preds = pd.DataFrame({
        "id": np.arange(len(df_test)),
        "relative_path": df_test["relative_path"],
        "id_grupo": df_test["id_grupo"],
        "classe_nome": df_test["class"],
        "y_true": y_test.flatten().astype(int),
        "y_prob": test_probs
    })
    df_preds.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPredicoes guardadas em: {PREDICTIONS_CSV} (Total: {len(df_preds)} linhas)")

    # Calcular ROC-AUC do teste independente de limiar
    y_true_test = df_preds["y_true"].values
    y_prob_test = df_preds["y_prob"].values
    test_auc = roc_auc_score(y_true_test, y_prob_test)
    print(f"\nROC-AUC no conjunto de Teste: {test_auc:.4f}")

    # Curva ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true_test, y_prob_test)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Aleatorio (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Sensibilidade)")
    plt.title("Curva ROC — Avaliacao no Conjunto de Teste (Aula 4)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PNG, dpi=300)
    plt.close()
    print(f"Curva ROC guardada com sucesso em: {ROC_CURVE_PNG}")


def evaluate_threshold_on_test(threshold):
    """Le as predicoes diretamente de aula4/results/test_predictions.csv SEM chamar o modelo."""
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(f"Ficheiro de predicoes nao encontrado: {PREDICTIONS_CSV}")

    df_preds = pd.read_csv(PREDICTIONS_CSV)
    y_true = df_preds["y_true"].values
    y_prob = df_preds["y_prob"].values

    preds = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds)
    total_samples = cm.sum()
    print("\n" + "=" * 70)
    print(f"AVALIACAO NO TESTE COM LIMIAR = {threshold}")
    print("=" * 70)
    print(f"Matriz de Confusao (linhas: Real [0=Uninfected, 1=Parasitized], colunas: Previsto):")
    print(cm)
    print(f"Soma dos elementos da matriz de confusao: {total_samples} (Esperado: 4071)")
    assert total_samples == 4071, f"Erro: soma da matriz deu {total_samples}, esperado 4071"

    tn, fp, fn, tp = cm.ravel()
    print(f"  - Verdadeiros Negativos (Uninfected corretos): {tn}")
    print(f"  - Falsos Positivos (Uninfected classificados como Parasitized): {fp}")
    print(f"  - Falsos Negativos (Parasitized perdidos / falso alarme negativo): {fn}")
    print(f"  - Verdadeiros Positivos (Parasitized detetados): {tp}")

    print("\nClassification Report:")
    report = classification_report(
        y_true, preds,
        target_names=["Uninfected (0)", "Parasitized (1)"],
        digits=4
    )
    print(report)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Avaliacao no Teste - Aula 4")
    parser.add_argument("--step", choices=["run_all", "eval_threshold"], default="run_all")
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar a aplicar")
    args = parser.parse_args()

    if args.step == "eval_threshold":
        evaluate_threshold_on_test(args.threshold)
        return

    # Passo 1: Construir cache de teste
    X_test, y_test, df_test = build_test_cache()

    # Passo 2: Carregar modelo
    print(f"A carregar modelo final de {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo carregado. Total parametros: {model.count_params():,}")

    # Passo 3: Tabela de limiares no VAL (para escolha do utilizador)
    compute_val_threshold_table(model)

    # Passo 4: Predicao unica no teste, gravacao de CSV e ROC AUC
    run_test_inference_and_metrics(model, X_test, y_test, df_test)

    # Passo 5: Avaliar por defeito com limiar 0.5 apenas para referencia imediata
    evaluate_threshold_on_test(0.5)


if __name__ == "__main__":
    main()
