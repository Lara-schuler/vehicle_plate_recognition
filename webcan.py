import sys
import cv2
from openalpr import Alpr
import os
import re

# Inicializar OpenALPR
alpr = Alpr("br", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    print("Erro ao carregar OpenALPR")
    sys.exit(1)

alpr.set_top_n(10)  # Ajuste para focar nos 10 principais resultados
alpr.set_default_region("br")
conf_threshold = 70  # Limite de confiança mínima para aceitar a leitura

# Criar pasta para salvar as placas recortadas
output_dir = 'recortes_placas'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Expressão regular para validar formato de placas brasileiras (antigas e Mercosul)
placa_pattern = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

# Função para pré-processamento da imagem
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Redução de ruído com GaussianBlur
    return blur

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Ler o frame da webcam
    ret, img = cap.read()

    if not ret:
        print("Falha ao capturar imagem")
        break

    # Verifique o tamanho original da imagem
    original_img = img.copy()

    # Pré-processamento da imagem
    processed_img = preprocess_image(img)
    img_str = cv2.imencode('.jpg', processed_img)[1].tobytes()

    # Reconhecer a placa usando OpenALPR
    results = alpr.recognize_array(img_str)

    for plate in results['results']:
        detected_plate = plate['plate']
        confidence = plate['confidence']

        # Verificar se o texto detectado segue o padrão de placas brasileiras e se a confiança é alta
        if placa_pattern.match(detected_plate) and confidence > conf_threshold:

            x1 = plate['coordinates'][0]['x']
            y1 = plate['coordinates'][0]['y']
            x2 = plate['coordinates'][2]['x']
            y2 = plate['coordinates'][2]['y']

            # Adicionar padding e garantir que os limites não extrapolem a imagem
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)

            # Recortar imagem da placa
            plate_img = original_img[y1:y2, x1:x2]
            cv2.imwrite(f'{output_dir}/plate_img_{x1}_{y1}.jpg', plate_img)  # Salvar placa para verificar

            # Exibir a placa detectada
            print(f"Placa detectada: {detected_plate} Confiança: {confidence:.2f}")

            # Desenhar retângulo ao redor da placa
            cv2.putText(img, f"Placa: {detected_plate} Conf: {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Exibir imagem
    cv2.imshow('Placa Detectada', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
alpr.unload()
