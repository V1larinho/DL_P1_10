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

os.makedirs("aula3/modelos", exist_ok=True)
os.makedirs("aula3/logs", exist_ok=True)
os.makedirs("aula3/results", exist_ok=True)

def pad_and_resize(img_path, target_size=(224, 224)):
    with Image.open(img_path) as img:
        w, h = img.size
        max_dim = max(w, h)
        delta_w = max_dim - w
        delta_h = max_dim - h
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img_padded = ImageOps.expand(img, padding, fill=0)
        img_resized = img_padded.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img_resized)

class MalariaSequence(keras.utils.Sequence):
    def __init__(self, df, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
        
    def __getitem__(self, i):
        batch_idx = self.indices[i*self.batch_size:(i+1)*self.batch_size]
        batch_df = self.df.iloc[batch_idx]
        
        X = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        y = np.empty((len(batch_df), 1), dtype=np.float32)
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            img_path = os.path.join("data/raw", row["relative_path"])
            img = pad_and_resize(img_path, self.target_size)
            X[j] = img
            y[j] = 1.0 if row["class"] == "Parasitized" else 0.0
            
        return X, y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

print("A carregar paths do CSV...")
df_train = pd.read_csv("splits/train.csv")
df_val = pd.read_csv("splits/val.csv")

train_seq = MalariaSequence(df_train, batch_size=32, shuffle=True)
val_seq = MalariaSequence(df_val, batch_size=32, shuffle=False)

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
