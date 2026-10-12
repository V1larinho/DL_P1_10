import json
import codecs

nb_path = r'c:\Users\vilar\Desktop\Mestrado\deeeplearning\aula2\aula2_P1.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## 2.5 Data Augmentation\n', 'Modelo com RandomFlip, RandomRotation e RandomContrast.']
}

code_cell_1 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast\n',
        '\n',
        'inputs_aug = keras.Input(shape=(180, 180, 3))\n',
        'x = RandomFlip("horizontal_and_vertical", seed=42)(inputs_aug)\n',
        'x = RandomRotation(0.25, seed=42)(x)\n',
        'x = RandomContrast(0.2, seed=42)(x)\n',
        'x = layers.Rescaling(1./255)(x)\n',
        '\n',
        'for filters in [32, 64, 128, 256]:\n',
        '    x = layers.Conv2D(filters, 3, activation="relu")(x)\n',
        '    x = layers.MaxPooling2D(2)(x)\n',
        'x = layers.GlobalAveragePooling2D()(x)\n',
        'outputs_aug = layers.Dense(2, activation="softmax")(x)\n',
        '\n',
        'model_aug = keras.Model(inputs_aug, outputs_aug)\n',
        'model_aug.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])\n',
        '\n',
        'model_aug.summary()\n'
    ]
}

code_cell_2 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import time\n',
        'start_time = time.time()\n',
        'history_aug = model_aug.fit(\n',
        '    train_ds,\n',
        '    validation_data=val_ds,\n',
        '    epochs=30\n',
        ')\n',
        'end_time = time.time()\n',
        'print(f"Tempo de treino (Augmentation): {end_time - start_time:.2f} segundos")\n'
    ]
}

code_cell_3 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import pandas as pd\n',
        'import matplotlib.pyplot as plt\n',
        '\n',
        'history_aug_df = pd.DataFrame(history_aug.history)\n',
        'history_aug_df.to_csv("history_aug.csv", index=False)\n',
        '\n',
        'plt.figure(figsize=(12, 5))\n',
        'plt.subplot(1, 2, 1)\n',
        'plt.plot(history_aug.history["loss"], label="Treino")\n',
        'plt.plot(history_aug.history["val_loss"], label="Validação")\n',
        'plt.title("Loss (Data Augmentation)")\n',
        'plt.legend()\n',
        '\n',
        'plt.subplot(1, 2, 2)\n',
        'plt.plot(history_aug.history["accuracy"], label="Treino")\n',
        'plt.plot(history_aug.history["val_accuracy"], label="Validação")\n',
        'plt.title("Accuracy (Data Augmentation)")\n',
        'plt.legend()\n',
        '\n',
        'plt.savefig("curvas_aug.png")\n',
        'plt.show()\n'
    ]
}

nb['cells'].extend([markdown_cell, code_cell_1, code_cell_2, code_cell_3])

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cells appended successfully.')
