import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Passo 1: Importar as bibliotecas necessárias

# Passo 2: Carregar a arquitetura pré-treinada (MobileNetV2)
model = MobileNetV2(weights='imagenet')

# Passo 3: Carregar a imagem que deseja identificar
img_path = 'caminho_para_a_imagem.jpg'  # Insira o caminho da imagem que deseja identificar
img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar a imagem para o tamanho esperado pelo modelo (224x224)

# Passo 4: Pré-processar a imagem e convertê-la em um array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Passo 5: Fazer a predição usando o modelo
preds = model.predict(x)

# Passo 6: Decodificar as predições em uma lista de tuplas (classe, descrição, probabilidade)
decoded_preds = decode_predictions(preds, top=3)[0]

# Passo 7: Exibir as predições
for pred in decoded_preds:
    print(f'Classe: {pred[1]}, Descrição: {pred[2]*100:.2f}%')