# Transfer Learning & Fine-Tuning – Classificador de Imagens

Aprenda a treinar um classificador de imagens **preciso** usando **poucos exemplos**, reaproveitando o conhecimento de uma rede neural pré-treinada (ex.: **VGG16** treinada no ImageNet). Este README resume conceitos, estratégias e traz um **exemplo prático** (Keras/TensorFlow) de **feature extraction** e **fine-tuning**.

> **O que é transfer learning?**  
> É aproveitar uma rede já treinada em um grande dataset e usá-la como base para um novo problema, acelerando o treinamento e melhorando a acurácia quando temos **poucos dados**.

---

## Sumário

- [Visão geral](#visão-geral)
- [Estratégias](#estratégias)
  - [Feature extraction](#feature-extraction)
  - [Fine-tuning](#fine-tuning)
- [Estrutura de dados](#estrutura-de-dados)
- [Requisitos](#requisitos)
- [Exemplo rápido (Keras/TensorFlow)](#exemplo-rápido-kerastensorflow)
  - [Baseline do zero (opcional)](#baseline-do-zero-opcional)
  - [Transfer learning com VGG16 (feature-extraction)](#transfer-learning-com-vgg16-feature-extraction)
  - [Fine-tuning (descongelando camadas finais)](#fine-tuning-descongelando-camadas-finais)
- [Resultados esperados](#resultados-esperados)
- [Boas práticas](#boas-práticas)
- [Referências e leituras](#referências-e-leituras)
- [Licença](#licença)

---

## Visão geral

Treinar uma rede “do zero” com um dataset pequeno tende a gerar **overfitting** e baixa acurácia. Com **transfer learning**:

- Reaproveitamos **extratores de características** já aprendidos (bordas, texturas, formas).
- Treinamos **menos parâmetros**, com **taxa de aprendizado menor**.
- Obtemos **ganhos substanciais** de desempenho com poucos dados (ex.: centenas de imagens).

---

## Estratégias

### Feature extraction
- **Congelar** todas as camadas do backbone pré-treinado (ex.: VGG16) e **substituir a cabeça de classificação** pela sua (com o nº de classes do seu problema).
- Treinar **apenas** a nova cabeça (rápido, estável).

### Fine-tuning
- Partimos do pré-treinado e **descongelamos** algumas camadas finais (as mais específicas).
- Ajustamos com **learning rate menor**.
- Útil quando seu dataset é razoável e **parecido** com o dataset original (ImageNet).

> Na prática, você pode fazer algo **híbrido**: congelar camadas iniciais (genéricas) e ajustar apenas as finais.

---

## Estrutura de dados

Use a convenção de diretórios do Keras `image_dataset_from_directory`:

```
data/
  train/
    classe_1/ img001.jpg ...
    classe_2/ ...
    ...
  val/
    classe_1/ ...
    classe_2/ ...
    ...
```

> Renomeie as pastas conforme suas classes. Funciona bem com centenas a poucos milhares de imagens.

---

## Requisitos

- Python 3.9+
- TensorFlow 2.15+ (CPU ou GPU)
- (Opcional) NVIDIA CUDA/cuDNN para acelerar no GPU

Instalação rápida:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install tensorflow matplotlib
```

---

## Exemplo rápido (Keras/TensorFlow)

> Ajuste `DATA_DIR` e `NUM_CLASSES` ao seu projeto.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10

train_ds = keras.utils.image_dataset_from_directory(
    Path(DATA_DIR) / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True,
)
val_ds = keras.utils.image_dataset_from_directory(
    Path(DATA_DIR) / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
)

NUM_CLASSES = len(train_ds.class_names)
```

### Baseline do zero (opcional)

Treinar um modelo pequeno do zero, para comparação:

```python
def build_baseline(num_classes):
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

baseline = build_baseline(NUM_CLASSES)
baseline.compile(optimizer="adam",
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])
baseline.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

### Transfer learning com VGG16 (feature extraction)

```python
base = keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)
base.trainable = False  # congela TODAS as camadas

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = keras.applications.vgg16.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fe = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

### Fine-tuning (descongelando camadas finais)

Após algumas épocas de **feature extraction**, podemos ajustar as últimas camadas:

```python
# Descongele parcialmente o backbone (ex.: último bloco convolucional)
for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-12:]:  # ajuste o fatiamento conforme necessário
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # menor LR
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=max(5, EPOCHS//2))
```

> **Dica:** Comece com `base.trainable = False`. Quando a validação estabilizar, libere **poucas** camadas finais e reduza a LR.

---

## Resultados esperados

- Em datasets com ~**6.000 imagens** e **97 classes**, é comum chegar a **~80%** de acurácia (varia por domínio e balanceamento).
- Com **pouquíssimos dados**, transfer learning geralmente supera o treino do zero.

---

## Boas práticas

- **Balanceie** as classes (ou use técnicas de balanceamento/aug).
- Use **data augmentation** moderada (flip, rotate, color jitter).
- **Early stopping** + **checkpoint** do melhor modelo.
- **LR menor** em fine-tuning.
- Congele **camadas iniciais** (genéricas) e ajuste as **finais** (específicas).

---

## Referências e leituras

- **VGG16**: arquitetura clássica vencedora do ILSVRC 2014.  
- **ImageNet**: dataset amplo usado no pré-treino.  
- **Transfer Learning & Fine-Tuning**: guias da documentação do TensorFlow/Keras e PyTorch.

> A estratégia ideal depende de: **tamanho do dataset**, **nº de classes** e **similaridade** com o dataset de pré-treino.

---

## Licença

Este projeto/README está sob licença **MIT**. Sinta-se à vontade para usar, adaptar e compartilhar.

---

### ⭐ Dê um star!
Se este README te ajudou, deixe uma ⭐ no repositório. Contribuições e PRs são bem-vindos!
