import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
"""# Cell 1: Setup
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import torch
import keras
from keras import layers
import random
import time
from PIL import Image, ImageOps

SEED = 42
keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("modelos", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("cache", exist_ok=True)
""",

"""# Cell 2: Data Loader with Caching logic
def pad_and_resize(img_path, target_size=(224, 224)):
    with Image.open(img_path) as img:
        w, h = img.size
        max_dim = max(w, h)
        delta_w = max_dim - w
        delta_h = max_dim - h
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img_padded = ImageOps.expand(img, padding, fill=0)
        img_resized = img_padded.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img_resized, dtype=np.uint8)

def build_cache(csv_path, X_cache_path, y_cache_path):
    if os.path.exists(X_cache_path) and os.path.exists(y_cache_path):
        print(f"Cache encontrada: {X_cache_path}")
        return np.load(X_cache_path), np.load(y_cache_path)
        
    print(f"A construir cache para {csv_path}...")
    df = pd.read_csv(csv_path)
    n = len(df)
    X = np.empty((n, 224, 224, 3), dtype=np.uint8)
    y = np.empty((n, 1), dtype=np.float32)
    
    for i, (_, row) in enumerate(df.iterrows()):
        img_path = os.path.join("../data/raw", row["relative_path"])
        X[i] = pad_and_resize(img_path)
        y[i] = 1.0 if row["class"] == "Parasitized" else 0.0
            
    np.save(X_cache_path, X)
    np.save(y_cache_path, y)
    print("Cache guardada!")
    return X, y

print("A carregar/construir cache...")
X_train, y_train = build_cache("../splits/train.csv", "cache/train_X_224.npy", "cache/train_y.npy")
X_val, y_val = build_cache("../splits/val.csv", "cache/val_X_224.npy", "cache/val_y.npy")

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
        X_batch = self.X[batch_idx].astype(np.float32) # Convert from uint8 0-255 to float32 0-255
        y_batch = self.y[batch_idx]
        return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_seq = CacheSequence(X_train, y_train, batch_size=32, shuffle=True)
val_seq = CacheSequence(X_val, y_val, batch_size=32, shuffle=False)
""",

"""# Cell 3: Build Model
# Confirm that inputs are 0-255 float
X_batch, y_batch = train_seq[0]
print(f"Batch min: {X_batch.min()}, max: {X_batch.max()} (esperado 0-255 float32)")

inputs = keras.Input(shape=(224, 224, 3))

# Data Augmentation (only active during training)
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.2)(x)

# EfficientNetB0 expects 0-255
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

print("Resumo da Arquitetura:")
model.summary()

# Conta parametros treinaveis
trainable_params = sum(np.prod(p.shape) for p in model.trainable_weights)
print(f"Parametros treinaveis: {trainable_params}")
""",

"""# Cell 4: Compile and Train
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("modelos/fe_melhor.keras", monitor="val_loss", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="logs/fe")
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
print(f"Treino concluído em {total_time:.2f} segundos.")
""",

"""# Cell 5: Resultados
best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_loss = history.history["val_loss"][best_epoch - 1]
best_val_auc = history.history["val_auc"][best_epoch - 1]
best_val_acc = history.history["accuracy"][best_epoch - 1] if "accuracy" in history.history else history.history["val_accuracy"][best_epoch - 1]

print(f"Melhor epoca: {best_epoch} com val_loss = {best_val_loss:.4f}")

# Guardar em CSV
resultados = pd.DataFrame([{
    "Modelo": "EfficientNetB0 (Feature Extraction)",
    "Trainable Params": trainable_params,
    "Best Epoch": best_epoch,
    "Val Loss": best_val_loss,
    "Val AUC": best_val_auc,
    "Val Accuracy": best_val_acc,
    "Time (s)": round(total_time, 2)
}])

resultados.to_csv("results/fe_results.csv", index=False)
display(resultados)
"""
]

for code in code_cells:
    nb.cells.append(nbf.v4.new_code_cell(code))

os.makedirs("aula3", exist_ok=True)
with open("aula3/aula3_P1.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook aula3_P1.ipynb atualizado com sucesso.")
