import os
import sys

# Sem buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
import torch
import keras
from keras import layers
import time

print(f"GPU Disponivel: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

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

print("\nA testar 1 epoca de Fine-Tuning para medir tempo...")
t0 = time.time()
h = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=1
)
t1 = time.time()
elapsed = t1 - t0

print(f"\nTEMPO_1_EPOCA: {elapsed:.2f} segundos ({elapsed/60:.2f} minutos)")
print(f"Estimativa para 20 epocas completas: {(elapsed * 20) / 60:.1f} minutos")
print(f"Resultados da 1a epoca: Loss={h.history['loss'][0]:.4f}, Acc={h.history['accuracy'][0]:.4f}, Val_Loss={h.history['val_loss'][0]:.4f}, Val_Acc={h.history['val_accuracy'][0]:.4f}, Val_AUC={h.history['val_auc'][0]:.4f}")
