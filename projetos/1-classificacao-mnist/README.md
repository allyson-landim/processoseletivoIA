# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Allyson Andre Almeida de Castro Paes Landim

### 1️⃣ Resumo da Arquitetura do Modelo
A arquitetura da CNN foi estruturada com o objetivo de extrair características mantendo um baixo custo computacional. A rede é composta por 3 blocos convolucionais sucessivos. Cada bloco utiliza uma camada `Conv2D` (com 32 filtros no primeiro e 64 nos subsequentes, ambas com ativação ReLU), seguida de `BatchNormalization` para estabilizar os gradientes e acelerar a convergência, e `MaxPooling2D` para redução de dimensionalidade espacial. 

Após a extração de features, os dados são planificados (`Flatten`) e submetidos a uma camada de `Dropout` com taxa de 50% para atuar como regularização e mitigar o overfitting. A camada de saída é densa, contendo 10 neurônios (um para cada classe) com ativação Softmax.

Para a validação, separamos de forma explícita 20% do conjunto de treinamento (`validation_split=0.2`). O treinamento foi gerenciado por uma estratégia de `EarlyStopping`, monitorando a perda de validação (`val_loss`) com uma paciência de 3 épocas e restaurando os melhores pesos ao final.

### 2️⃣ Bibliotecas Utilizadas
*   **TensorFlow / Keras** (usado para processamento do dataset MNIST, construção, treinamento e inferência Edge com tf.lite)
*   **NumPy** (usado para manipulações matriciais e pré-processamento de dados na inferência)

### 3️⃣ Técnica de Otimização do Modelo
A técnica aplicada no `optimize_model.py` foi a **Dynamic Range Quantization** (Quantização de Faixa Dinâmica) nativa do conversor do TensorFlow Lite (`tf.lite.Optimize.DEFAULT`). Essa técnica quantiza os pesos do modelo (que originalmente são armazenados em ponto flutuante de 32 bits - `float32`) para inteiros de 8 bits (`int8`) pós-treinamento. O resultado é uma redução no tamanho do modelo em quase 4 vezes, o que é crucial para deploy em dispositivos Edge com memória restrita, sofrendo uma degradação quase nula na acurácia.

### 4️⃣ Resultados Obtidos
*   **Acurácia de Validação:** 99.01%
*   **Tamanho do `model.h5`:** 733KB
*   **Tamanho do `model.tflite`:** 67.6KB

### 5️⃣ Comentários Adicionais (Opcional)
A maior decisão técnica do projeto consistiu no balanceamento do modelo: ele precisava ser profundo o suficiente para atingir uma precisão satisfatória (cerca de 99%), mas leve o suficiente para ser considerado eficiente em um fluxo de Edge AI. A utilização de Batch Normalization eliminou a necessidade de treinamentos muito longos, permitindo que a rede convergisse rapidamente dentro das restrições de treinamento exclusivo por CPU. 

### 6️⃣ Exemplo de Inferência
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Todas as 5 amostras iniciais foram preditas corretamente, indicando que a quantização dos pesos de float32 para int8 não prejudicou o desempenho de inferência em casos reais.
