# Transfer Learning / Fine-Tuning (PT-BR) — Notebook Original + Ajustes para CPU

Este repositório traz o **notebook original** de Transfer Learning com Keras/TensorFlow, mantendo o conteúdo **intacto**, e adiciona **apenas**:
1. Uma seção **`introducao_br`** (PT-BR) no topo do notebook, explicando objetivos, como interpretar métricas e dicas de prática.
2. Duas melhorias **para CPU** marcadas como **`[ALTERAÇÃO CPU]`**:
   - Forçar backend do Keras para **TensorFlow** e **ocultar GPUs** (inclusive DirectML no Windows) para execução estável em **CPU**.
   - Aplicar **`prefetch(tf.data.AUTOTUNE)`** nos datasets, se já estiverem criados, para acelerar I/O em CPU.

> Nenhuma célula original foi removida ou sobrescrita. As mudanças são **adições** e estão claramente comentadas.

---

## 🧠 O que o notebook faz
- Carrega um **modelo pré-treinado** (ex.: `VGG16`) do Keras Applications.
- **Feature extraction**: congela o backbone e treina apenas a cabeça de classificação.
- **Fine-tuning (opcional)**: libera um pequeno conjunto de camadas finais com **learning rate** reduzida.
- Exibe logs por época: `accuracy`, `val_accuracy`, `loss`, `val_loss` e (em geral) plota curvas para análise.

### Como interpretar os resultados
- **Acurácia** (`accuracy`, `val_accuracy`): proporção de acertos no **treino** e **validação**.
- **Perda** (`loss`, `val_loss`): valor da função de custo. O ideal é que **ambas** diminuam.
- **Overfitting**: se `loss` (treino) ↓ mas `val_loss` ↑ de forma consistente, ajuste regularização/augmentation, reduza fine-tuning ou **pare antes** (EarlyStopping).
- **Transfer learning** é especialmente útil com poucos dados — os primeiros *epochs* já devem mostrar melhora de `val_accuracy`.

---

## 🚀 Como rodar (Google Colab ou local)

### Opção A — Google Colab (recomendado para começar)
1. Envie o notebook **`transfer_learning_ptbr_cpu.ipynb`** para o Colab.
2. (Se necessário) Instale dependências no início:
   ```python
   !pip -q install "tensorflow==2.17.*" "keras>=3,<4" "numpy==1.26.4" "matplotlib"
   ```
3. Execute as células na ordem. As duas células de **`[ALTERAÇÃO CPU]`** estão no topo.

### Opção B — Local (Windows/Conda)
```powershell
conda create -y -n tf311 -c conda-forge python=3.11 jupyterlab notebook sqlite
conda activate tf311
python -m pip install --upgrade pip
python -m pip install -r requirements_cpu.txt

# evitar conflito com pacotes do User Site (opcional, recomendado)
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate && conda activate tf311

# registrar kernel no Jupyter
python -m ipykernel install --user --name tf311 --display-name "Python (tf311)"
jupyter notebook
```
No notebook, selecione o kernel **Python (tf311)**.  
Confirme dentro da primeira célula: `TF: 2.17.*` e a lista de dispositivos **sem GPU** (CPU forçada).

---

## 🧩 Dataset
O notebook original foi escrito para **imagens** e utiliza pré-processamento compatível com `VGG16`.  
Você pode usar:
- **MNIST** (conforme o link do desafio).
- **Cats vs Dogs** (binário).
- **Seu próprio dataset** de 2 classes (ex.: seus pets, objetos, etc.).

> Se usar pastas no formato `train/` e `val/`, ajuste os trechos de carregamento conforme o original (ex.: `ImageDataGenerator` ou APIs equivalentes do Keras).

---

## ⚙️ Ajustes de desempenho (CPU)
- **Batch**: use 8–32. Batches maiores em CPU podem **piorar** o tempo/epoch.
- **Imagem**: 160–224 px (conforme o modelo pré-treinado escolhido).
- **Prefetch**: já incluído pela alteração **`[ALTERAÇÃO CPU]`** (só se os datasets existirem).
- **Fine-tuning**: comece **desligado** (só feature extraction) e ative **poucas** camadas finais quando a validação **estabilizar**.
- **Callbacks**: use `EarlyStopping` e `ModelCheckpoint` quando possível.

---

## 🧪 Métricas & Diagnóstico
- Acompanhe `val_accuracy` e `val_loss`. Se ficarem **estáveis** ou piorarem, ajuste LR/camadas/augmentation.
- Compare **baseline do zero** vs **transfer learning**: o TL tende a convergir mais rápido e com melhor `val_accuracy` em poucos dados.

---

## 📦 Arquivos do repositório
```
.
├── README.md
├── requirements_cpu.txt
└── transfer_learning_ptbr_cpu.ipynb
```
- `transfer_learning_ptbr_cpu.ipynb`: notebook original + **`introducao_br`** + **`[ALTERAÇÃO CPU]`**.
- `requirements_cpu.txt`: versões pinadas para evitar conflitos (NumPy 1.26.x, TF 2.17.x, Keras 3).

---

## ✅ Checklist (DIO)
- [ ] Executar o notebook e documentar **sua experiência** (o que mudou, resultados, prints).
- [ ] Preencher este `README.md` com **prints** (pasta `/images`) e **observações pessoais**.
- [ ] Publicar o repositório no GitHub e enviar o link na plataforma.
- [ ] (Opcional) Usar seu próprio dataset de 2 classes.

---

## 📜 Licença
MIT — sinta-se à vontade para usar, adaptar e compartilhar.
