import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
import torch
import keras
from keras import layers
import random
import time

print(f"GPU Disponivel: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

SEED = 42
keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("aula3/modelos", exist_ok=True)
os.makedirs("aula3/logs/ft", exist_ok=True)
os.makedirs("aula3/results", exist_ok=True)

print("A carregar cache...")
X_train = np.load("aula3/cache/train_X_224.npy")
y_train = np.load("aula3/cache/train_y.npy")
X_val = np.load("aula3/cache/val_X_224.npy")
y_val = np.load("aula3/cache/val_y.npy")
print(f"Cache carregada. Train: {X_train.shape}, Val: {X_val.shape}")

class CacheSequence(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
        
    def __getitem__(self, i):
        batch_idx = self.indices[i*self.batch_size:(i+1)*self.batch_size]
        X_batch = self.X[batch_idx].astype(np.float32)
        y_batch = self.y[batch_idx]
        return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_seq = CacheSequence(X_train, y_train, batch_size=32, shuffle=True)
val_seq = CacheSequence(X_val, y_val, batch_size=32, shuffle=False)

print("A carregar modelo fe_melhor.keras...")
model = keras.models.load_model("aula3/modelos/fe_melhor.keras")

params_before = sum(np.prod(p.shape) for p in model.trainable_weights)
print(f"Parametros treinaveis ANTES: {params_before:,}")

base = model.get_layer("efficientnetb0")
base.trainable = True

for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers[-20:]:
    layer.trainable = True

params_after = sum(np.prod(p.shape) for p in model.trainable_weights)
print(f"Parametros treinaveis DEPOIS (ultimas 20 camadas): {params_after:,}")

# Recompilar com learning rate 1e-5
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("aula3/modelos/ft_melhor.keras", monitor="val_loss", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="aula3/logs/ft")
]

print("A iniciar o treino de Fine-Tuning (max 20 epocas)...")
start_time = time.time()
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=20,
    callbacks=callbacks
)
end_time = time.time()
total_time = end_time - start_time
print(f"Treino de Fine-Tuning concluido em {total_time:.2f} s ({total_time/60:.2f} min)")

# Guardar history completo do FT
history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", range(1, len(history_df) + 1))
history_df.to_csv("aula3/results/history_ft.csv", index=False)
print("Guardado aula3/results/history_ft.csv")

# Extrair metricas da melhor epoca (menor val_loss)
best_epoch_idx = np.argmin(history.history["val_loss"])
best_epoch = best_epoch_idx + 1
best_val_loss = history.history["val_loss"][best_epoch_idx]
best_val_auc = history.history["val_auc"][best_epoch_idx]
best_val_acc = history.history["val_accuracy"][best_epoch_idx]

print(f"Melhor Epoca (FT): {best_epoch} (1-indexed)")
print(f"Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}, Val AUC: {best_val_auc:.4f}")

# Tabela comparativa com as 3 abordagens
tabela_comp = pd.DataFrame([
    {
        "Abordagem": "Baseline CNN (Aula 2)",
        "Parametros Treinaveis": 388930,
        "Melhor Epoca": 10,
        "Val Loss": 0.1335,
        "Val AUC": "N/A",
        "Val Accuracy": 0.9637,
        "Tempo Treino (s)": 847.0
    },
    {
        "Abordagem": "Feature Extraction (Aula 3)",
        "Parametros Treinaveis": 1281,
        "Melhor Epoca": 17,
        "Val Loss": 0.1486,
        "Val AUC": 0.9852,
        "Val Accuracy": 0.9503,
        "Tempo Treino (s)": 4861.62
    },
    {
        "Abordagem": "Fine-Tuning (Aula 3, 20 camadas)",
        "Parametros Treinaveis": params_after,
        "Melhor Epoca": best_epoch,
        "Val Loss": round(best_val_loss, 4),
        "Val AUC": round(best_val_auc, 4),
        "Val Accuracy": round(best_val_acc, 4),
        "Tempo Treino (s)": round(total_time, 2)
    }
])

tabela_comp.to_csv("aula3/results/tabela_comparativa.csv", index=False)
print("\nTabela comparativa guardada em aula3/results/tabela_comparativa.csv:")
print(tabela_comp.to_string(index=False))
print("\nFIM DO FINE-TUNING")
