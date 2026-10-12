import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
import torch
import keras
from keras import layers
import time
from PIL import Image, ImageOps

print(f"GPU Disponivel: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

os.makedirs("aula3/cache", exist_ok=True)
os.makedirs("aula3/logs", exist_ok=True)

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
    
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        img_path = os.path.join("data/raw", row["relative_path"])
        X[i] = pad_and_resize(img_path)
        y[i] = 1.0 if row["class"] == "Parasitized" else 0.0
        if (i+1) % 5000 == 0:
            print(f"  Processadas {i+1}/{n} imagens... ({(time.time()-t0):.1f}s)")
            
    print("A guardar no disco...")
    np.save(X_cache_path, X)
    np.save(y_cache_path, y)
    print(f"Cache guardada! Tempo total: {time.time()-t0:.1f}s")
    return X, y

X_train, y_train = build_cache("splits/train.csv", "aula3/cache/train_X_224.npy", "aula3/cache/train_y.npy")
X_val, y_val = build_cache("splits/val.csv", "aula3/cache/val_X_224.npy", "aula3/cache/val_y.npy")

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
        # Converter uint8 para float32 dinamicamente
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

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

print("A testar 1 epoca para medir tempo...")
t0 = time.time()
model.fit(train_seq, validation_data=val_seq, epochs=1)
t1 = time.time()

print(f"Tempo de 1 epoca: {t1 - t0:.2f} segundos")
print(f"Tempo estimado para 30 epocas: {((t1-t0)*30)/60:.1f} minutos")
