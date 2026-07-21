import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# Versão final do código do arquivo train_model.py

def create_model():
    """
    Constrói a arquitetura da Rede Neural Convolucional (CNN).
    A rede é composta por 3 blocos de extração de características (Convolução + Normalização + Pooling),
    seguidos por uma etapa de classificação com regularização.
    """
    model = keras.Sequential()
    
    # --- Bloco Convolucional 1 ---
    # Extrai os primeiros padrões do input de 28x28 com 1 canal (tons de cinza).
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # --- Bloco Convolucional 2 ---
    # Extrai padrões intermediários e mais complexos.
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # --- Bloco Convolucional 3 ---
    # Extrai características de alto nível.
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten para transformar o tensor em vetor 1D
    model.add(layers.Flatten())
    
    # Dropout para regularização (evitar overfitting)
    model.add(layers.Dropout(0.5))
    
    # Saída com 10 classes (dígitos de 0 a 9)
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

def main():
    # 1. Carregar o dataset MNIST via tf.keras.datasets.mnist
    print("Carregando o dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 2. Normaliza para [0, 1] e ajusta o shape para (28, 28, 1)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

    # 4. Construir a CNN
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 5. Configura o Early Stopping focado no val_loss
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    print("\nIniciando o treinamento (CPU)...")
    # Força o uso de CPU
    with tf.device('/CPU:0'):
        # 3. Faz o split de 20% para validação direto no fit
        history = model.fit(
            x_train, y_train,
            epochs=15,          # Limite máximo de épocas estabelecido
            batch_size=64,      # Atualiza os pesos a cada 64 imagens
            validation_split=0.2, 
            callbacks=[early_stopping]
        )

    # 6. Exibir a acurácia de validação final no terminal
    print("\nAvaliando o modelo no conjunto de validação final...")
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"\n==================================================")
    print(f">>> Acurácia de validação final: {val_accuracy * 100:.2f}% <<<")
    print(f"==================================================\n")

    # 7. Salvar o modelo treinado como "model.h5"
    model_path = 'model.h5'
    model.save(model_path)
    print(f"Modelo salvo com sucesso no diretório atual como: {model_path}")

if __name__ == '__main__':
    main()
