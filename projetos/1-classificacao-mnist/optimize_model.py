import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# Versão final do código do arquivo optimize_model.py

def main():
    # 1. Carrega o modelo Keras treinado anteriormente
    print("Carregando o modelo 'model.h5'...")
    model = tf.keras.models.load_model('model.h5')
    
    # 2. Inicializa o conversor do TensorFlow Lite usando o modelo carregado
    print("Inicializando o conversor TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. Aplica a técnica de otimização: Dynamic Range Quantization
    # A flag tf.lite.Optimize.DEFAULT habilita a quantização de pesos para 8 bits.
    print("Aplicando otimização (Dynamic Range Quantization)...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 4. Realiza a conversão do modelo
    print("Convertendo o modelo para TensorFlow Lite...")
    tflite_model = converter.convert()
    
    # 5. Salva o modelo otimizado como um arquivo binário (.tflite)
    tflite_model_path = 'model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Modelo otimizado salvo com sucesso em: {tflite_model_path}")

if __name__ == '__main__':
    main()
