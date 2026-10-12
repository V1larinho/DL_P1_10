# Aula Prática 2 — Resultados

## 2.1 — Caracterização do Dataset

| Classe | N.º imagens | % | Dimensão típica | Observações |
|--------|-------------|---|-----------------|-------------|
| Parasitized | 13779 | 50.0% | mediana 133×133 | dimensões variáveis (min 58×46, max 364×340). Sem marcas óbvias. |
| Uninfected | 13779 | 50.0% | mediana 130×130 | dimensões variáveis (min 64×49, max 226×229). Sem marcas óbvias. |

*Observações sobre dimensões:* Não há bordos pretos consistentes na grande maioria nem diferenças de tamanho que permitam distinguir as classes (as medianas são quase iguais).

**Total:** 27558 imagens  
**Grupos únicos:** 200  
**Imagens corrompidas:** 0  
**Duplicados (md5):** 0 imagens em 0 grupos

## 2.2 — Divisão sem fugas

**Método:** Reutilização dos splits do D1 (leitura direta dos ficheiros `train.csv`, `val.csv` e `test.csv`). No D1, a divisão original foi feita iterando sobre várias seeds no `GroupShuffleSplit` para otimizar a estratificação (não foi usada uma seed 42 fixa).  
**Justificação:** Garante consistência com o trabalho anterior e respeita a integridade de grupos.

| Verificação | Resultado |
|------------|----------|
| Interseção MD5 train∩val | 0 |
| Interseção MD5 train∩test | 0 |
| Interseção MD5 val∩test | 0 |
| Interseção grupo train∩val | 0 |
| Interseção grupo train∩test | 0 |
| Interseção grupo val∩test | 0 |

| Split | N | Parasitized % | Uninfected % |
|-------|---|--------------|-------------|
| train | 19380 | 50.1% | 49.9% |
| val | 4107 | 50.0% | 50.0% |
| test | 4071 | 49.4% | 50.6% |

## 2.3 — Baseline CNN

**Arquitetura:** Input(180,180,3) → Rescaling(1/255) → 4×[Conv2D(f,3,relu)+MaxPool(2)] com f∈[32,64,128,256] → GAP → Dense(2, softmax)  
**Optimizer:** Adam  
**Loss:** sparse_categorical_crossentropy  
**Épocas:** 30 (sem EarlyStopping)  
**Tempo de treino:** 847.0s (14.1 min)

Verificação de parâmetros:
- Conv2D #1: (3×3×3+1)×32 = 896 (✅)
- Conv2D #2: (3×3×32+1)×64 = 18496 (✅)

## 2.4 — Diagnóstico

| Métrica | Época 10 (menor val_loss) | Época 30 (final) |
|---------|---------------------------|------------------|
| Acc treino | 0.9537 (95.37%) | 0.9834 (98.34%) |
| Acc val | 0.9637 (96.37%) | 0.9530 (95.30%) |
| Loss treino | 0.1413 | 0.0459 |
| Loss val | 0.1334 | 0.2339 |

**Classificador maioritário:** 0.5004 (50.04%)  
**Diagnóstico:** Overfitting a partir da época 10.

**Justificação:** Até à época 10, o modelo generaliza muito bem (gap treino-val negativo na accuracy, que atinge 0.9637). Após a época 10, a `val_loss` começa a subir, atingindo 0.2339 na época 30 (um aumento de ~0.10), enquanto a loss de treino continua a descer quase para zero e a accuracy de treino sobe para 98.34%. O gap na época 30 passa a ser de +3.04%. Isto é um indicador claro de overfitting nas épocas finais.

## 2.5 — Data Augmentation

**Alterações:** Adição das camadas `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.25)` e `RandomContrast(0.2)` imediatamente após o `Input` (antes do `Rescaling`). Parâmetros de treino iguais.

### Tabela Comparativa (Baseline vs Data Augmentation)

| Métrica | Baseline | Data Augmentation |
|---------|----------|-------------------|
| Melhor Época (menor `val_loss`) | 10 | 24 |
| Menor `val_loss` | 0.1334 | 0.1117 |
| Acc Treino (melhor época) | 95.37% | 95.62% |
| Acc Validação (melhor época) | 96.37% | 96.54% |
| `val_loss` na Época 30 | 0.2339 | 0.1323 |
| Acc Treino na Época 30 | 98.34% | 95.88% |
| Acc Validação na Época 30 | 95.30% | 96.15% |
| Gap (Acc Treino - Acc Val) na Época 30 | +3.04% | -0.27% |

### Análise de Overfitting

Com base estritamente nos números, o **overfitting diminuiu drasticamente**. 
No baseline, a `val_loss` subia de 0.1334 para 0.2339 e existia um gap de +3.04% entre o treino e a validação na época 30. Com o Data Augmentation, o gap na época 30 é praticamente nulo (-0.27%, indicando que a precisão de validação acompanha perfeitamente o treino), e a `val_loss` na época 30 manteve-se baixa (0.1323).
Esta diferença é consistente e visível ao longo das épocas (ver curvas).

### Propostas da IA

| Proposta | Justificação |
|----------|--------------|
| | |
| | |
| | |

## Notas / O que correu mal

- Os modelos refletem uma execução única (sem repetições) fixada com seed 42 onde possível, mas sujeita a não-determinismo da GPU.
- O notebook correu sem erros ("Restart & Run All" validado).
- Nenhum outro problema encontrado.
