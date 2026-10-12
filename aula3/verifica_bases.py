import os
import sys

# Definir backend consistente com o ambiente
os.environ["KERAS_BACKEND"] = "torch"
import keras

print(f"Keras version: {keras.__version__}")
print(f"Backend: {keras.backend.backend()}\n")

models_to_test = [
    ("EfficientNetB0", keras.applications.EfficientNetB0, keras.applications.efficientnet),
    ("ResNet50", keras.applications.ResNet50, keras.applications.resnet50),
    ("ConvNeXtTiny", keras.applications.ConvNeXtTiny, keras.applications.convnext),
]

print("=" * 70)
print("1. CONTAGEM DE PARÂMETROS (sem descarregar pesos, weights=None)")
print("=" * 70)

for name, model_cls, module in models_to_test:
    print(f"\n--- {name} ---")
    # Sem topo (include_top=False)
    model_notop = model_cls(include_top=False, weights=None, input_shape=(224, 224, 3))
    params_notop = model_notop.count_params()
    print(f"  include_top=False: {params_notop:,} parâmetros ({params_notop})")
    
    # Com topo (include_top=True, classes=1000)
    model_top = model_cls(include_top=True, weights=None, classes=1000, input_shape=(224, 224, 3))
    params_top = model_top.count_params()
    print(f"  include_top=True : {params_top:,} parâmetros ({params_top})")
    print(f"  Diferença (cabeça 1000 classes): {params_top - params_notop:,} parâmetros")

print("\n" + "=" * 70)
print("2. DOCUMENTAÇÃO OFICIAL DO KERAS.APPLICATIONS SOBRE PREPROCESSAMENTO")
print("=" * 70)

for name, model_cls, module in models_to_test:
    print(f"\n{'='*25} {name} {'='*25}")
    doc = model_cls.__doc__ or ""
    # Procurar secções de inputs/preprocessing no docstring da classe
    print(f"Docstring da classe {name} (trecho relevante):")
    lines = doc.split("\n")
    # Imprimir as primeiras 35 linhas ou linhas com menção a input / preprocess / scale
    for line in lines[:40]:
        print("  ", line)
    
    # Verificar preprocess_input no respetivo modulo
    if hasattr(module, "preprocess_input"):
        prep_doc = module.preprocess_input.__doc__ or "Sem docstring"
        print(f"\n  preprocess_input em keras.applications.{module.__name__.split('.')[-1]}:")
        for line in prep_doc.strip().split("\n")[:20]:
            print("    ", line)
    else:
        print(f"\n  preprocess_input: Módulo {module.__name__} NÃO tem preprocess_input!")

print("\n" + "=" * 70)
print("VERIFICAÇÃO CONCLUÍDA")
print("=" * 70)
