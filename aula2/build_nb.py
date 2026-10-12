import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb.cells = []

# Title and Setup
nb.cells.append(nbf.v4.new_markdown_cell("# Aula 2 - P1\n\n## 2.1 Caracterização"))

setup_code = """
import os
import glob
import re
import hashlib
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from PIL import Image

# Fix seeds
os.environ['PYTHONHASHSEED'] = '42'
tf.random.set_seed(42)
np.random.seed(42)
import random
random.seed(42)

DATA_DIR = r"c:\\Users\\vilar\\Desktop\\Mestrado\\deeeplearning\\data\\raw\\cell_images"

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

data = []
for ext in ('*.png', '*.jpg', '*.jpeg'):
    for path in Path(DATA_DIR).rglob(ext):
        label = path.parent.name
        # extract group using regex
        match = re.search(r'^(C\d+(?:P\d+)?)', path.name)
        group = match.group(1) if match else "Unknown"
        data.append({"path": str(path), "label": label, "grupo": group, "filename": path.name})

df = pd.DataFrame(data)

# Value counts
print("Value counts absolutos:")
print(df['label'].value_counts())
print("\\nValue counts (%):")
print(df['label'].value_counts(normalize=True) * 100)
"""
nb.cells.append(nbf.v4.new_code_cell(setup_code))

char_code = """
widths = []
heights = []
md5s = []
corrupted = []

for p in df['path']:
    try:
        with Image.open(p) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
        md5s.append(compute_md5(p))
    except Exception as e:
        widths.append(np.nan)
        heights.append(np.nan)
        md5s.append(None)
        corrupted.append(p)

df['width'] = widths
df['height'] = heights
df['md5'] = md5s

print(f"Imagens corrompidas: {len(corrupted)}")
duplicates = df.duplicated(subset=['md5'], keep=False)
print(f"Duplicados (md5 igual): {duplicates.sum()}")

for label, group in df.groupby('label'):
    print(f"\\nClasse {label}:")
    print(f"Mediana: {group['width'].median()} x {group['height'].median()}")
    print(f"Min: {group['width'].min()} x {group['height'].min()}")
    print(f"Max: {group['width'].max()} x {group['height'].max()}")
"""
nb.cells.append(nbf.v4.new_code_cell(char_code))

nb.cells.append(nbf.v4.new_markdown_cell("## 2.2 Divisão sem fugas\nReaproveitando a divisão feita no D1, uma vez que já garantiu a estratificação 70/15/15 por classe e por grupo."))

split_code = """
D1_SPLITS_DIR = r"c:\\Users\\vilar\\Desktop\\Mestrado\\deeeplearning\\splits"
df_train = pd.read_csv(os.path.join(D1_SPLITS_DIR, "train.csv"))
df_val = pd.read_csv(os.path.join(D1_SPLITS_DIR, "val.csv"))
df_test = pd.read_csv(os.path.join(D1_SPLITS_DIR, "test.csv"))

df_train['split'] = 'train'
df_val['split'] = 'val'
df_test['split'] = 'test'

df_splits = pd.concat([df_train, df_val, df_test], ignore_index=True)

df_splits['path'] = df_splits['relative_path'].apply(
    lambda x: os.path.join(r"c:\\Users\\vilar\\Desktop\\Mestrado\\deeeplearning\\data\\raw", os.path.normpath(x))
)

# Merge para obter md5 e as outras informacoes (grupo tb ja vinha, mas validamos se eh o mesmo)
df_splits = df_splits.merge(df[['path', 'md5']], on='path', how='left')
df_splits.rename(columns={'class': 'label', 'id_grupo': 'grupo'}, inplace=True)
df_splits_out = df_splits[['path', 'label', 'grupo', 'md5', 'split']]
df_splits_out.to_csv("splits.csv", index=False)

# Checks
md5_train = set(df_splits_out[df_splits_out['split']=='train']['md5'].dropna())
md5_val = set(df_splits_out[df_splits_out['split']=='val']['md5'].dropna())
md5_test = set(df_splits_out[df_splits_out['split']=='test']['md5'].dropna())

print("Interseções de MD5:")
print("Train-Val:", len(md5_train.intersection(md5_val)))
print("Train-Test:", len(md5_train.intersection(md5_test)))
print("Val-Test:", len(md5_val.intersection(md5_test)))

grupo_train = set(df_splits_out[df_splits_out['split']=='train']['grupo'])
grupo_val = set(df_splits_out[df_splits_out['split']=='val']['grupo'])
grupo_test = set(df_splits_out[df_splits_out['split']=='test']['grupo'])

print("\\nInterseções de Grupo:")
print("Train-Val:", len(grupo_train.intersection(grupo_val)))
print("Train-Test:", len(grupo_train.intersection(grupo_test)))
print("Val-Test:", len(grupo_val.intersection(grupo_test)))

print("\\nDistribuição de classes por split:")
for split in ['train', 'val', 'test']:
    print(f"\\nSplit {split}:")
    print(df_splits_out[df_splits_out['split']==split]['label'].value_counts(normalize=True) * 100)
"""
nb.cells.append(nbf.v4.new_code_cell(split_code))

nb.cells.append(nbf.v4.new_markdown_cell("## 2.3 Baseline\nCriando as pastas e treinando a rede."))

baseline_code = """
import platform

split_base_dir = "split"
os.makedirs(split_base_dir, exist_ok=True)
for split in ['train', 'val', 'test']:
    for label in df_splits_out['label'].unique():
        os.makedirs(os.path.join(split_base_dir, split, label), exist_ok=True)

# criar as imagens
for _, row in df_splits_out.iterrows():
    src = row['path']
    dst = os.path.join(split_base_dir, row['split'], row['label'], os.path.basename(src))
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

img_size = (180, 180)
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(split_base_dir, 'train'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(split_base_dir, 'val'),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
    shuffle=False
)
"""
nb.cells.append(nbf.v4.new_code_cell(baseline_code))

train_code = """
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1./255)(inputs)
for filters in [32, 64, 128, 256]:
    x = layers.Conv2D(filters, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
"""
nb.cells.append(nbf.v4.new_code_cell(train_code))

math_code = """
print("Cálculo manual dos parâmetros:")
print("Conv 1 (32 filtros): (3 * 3 * 3 + 1) * 32 =", (3*3*3 + 1)*32)
print("Conv 2 (64 filtros): (3 * 3 * 32 + 1) * 64 =", (3*3*32 + 1)*64)
"""
nb.cells.append(nbf.v4.new_code_cell(math_code))

fit_code = """
import time

start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)
end_time = time.time()
print(f"Tempo de treino: {end_time - start_time:.2f} segundos")

val_labels = []
for images, labels in val_ds:
    val_labels.extend(labels.numpy())
majority_class_count = max(np.bincount(val_labels))
majority_acc = majority_class_count / len(val_labels)
print(f"Acurácia do classificador maioritário no val: {majority_acc:.4f}")
print(f"Acurácia final no val (última época): {history.history['val_accuracy'][-1]:.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(fit_code))

nb.cells.append(nbf.v4.new_markdown_cell("## 2.4 Curvas e diagnóstico"))

curves_code = """
history_df = pd.DataFrame(history.history)
history_df.to_csv('history_baseline.csv', index=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='train_loss')
plt.plot(history_df['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='train_accuracy')
plt.plot(history_df['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig('curvas_baseline.png')

best_epoch = history_df['val_loss'].idxmin()
print(f"Melhor época (menor val_loss): {best_epoch + 1}")
print(f"Acc treino na melhor época: {history_df.loc[best_epoch, 'accuracy']:.4f}")
print(f"Acc val na melhor época: {history_df.loc[best_epoch, 'val_accuracy']:.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(curves_code))

with open(r"c:\Users\vilar\Desktop\Mestrado\deeeplearning\aula2\aula2_P1.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
