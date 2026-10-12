import os
import sys

# Forçar stdout sem buffering para o log ir caindo em tempo real no ficheiro
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
os.makedirs("aula3/logs/fe", exist_ok=True)
os.makedirs("aula3/results", exist_ok=True)

print("A carregar cache...")
X_train = np.load("aula3/cache/train_X_224.npy")
y_train = np.load("aula3/cache/train_y.npy")
X_val = np.load("aula3/cache/val_X_224.npy")
y_val = np.load("aula3/cache/val_y.npy")
print("Cache carregada.")

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

inputs = keras.Input(shape=(224, 224, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.2)(x)
base = keras.applications.EfficientNetB0(
    include_top=False, 
    weights="imagenet", 
    input_shape=(224, 224, 3)
)
base.trainable = False
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

trainable_params = sum(np.prod(p.shape) for p in model.trainable_weights)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("aula3/modelos/fe_melhor.keras", monitor="val_loss", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="aula3/logs/fe")
]

print("A iniciar o treino...")
start_time = time.time()
history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=30,
    callbacks=callbacks
)
end_time = time.time()
total_time = end_time - start_time

best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_loss = history.history["val_loss"][best_epoch - 1]
best_val_auc = history.history["val_auc"][best_epoch - 1]
best_val_acc = history.history["accuracy"][best_epoch - 1] if "accuracy" in history.history else history.history["val_accuracy"][best_epoch - 1]

print(f"Melhor epoca: {best_epoch} com val_loss = {best_val_loss:.4f}")

resultados = pd.DataFrame([{
    "Modelo": "EfficientNetB0 (Feature Extraction)",
    "Trainable Params": trainable_params,
    "Best Epoch": best_epoch,
    "Val Loss": best_val_loss,
    "Val AUC": best_val_auc,
    "Val Accuracy": best_val_acc,
    "Time (s)": round(total_time, 2)
}])
resultados.to_csv("aula3/results/fe_results.csv", index=False)
print("FIM")
