import sys
import cv2
from openalpr import Alpr
import re
from vehicle_detection import detect_vehicles
from plate_recognition import recognize_plate
from utils import preprocess_image

# Inicializar OpenALPR
alpr = Alpr("br", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    print("Erro ao carregar OpenALPR")
    sys.exit(1)

alpr.set_top_n(10)  # Ajuste para focar nos 10 principais resultados
alpr.set_default_region("br")
conf_threshold = 50  # Limite de confiança mínima para aceitar a leitura

# Expressão regular para validar formato de placas brasileiras (antigas e Mercosul)
placa_pattern = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

# Configurações
cap = cv2.VideoCapture()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou falha ao carregar frame")
        break

    # Redimensionar o frame
    new_width = 800
    new_height = int(frame.shape[0] * (new_width / frame.shape[1]))
    frame = cv2.resize(frame, (new_width, new_height))
     # Pré-processar a imagem antes da detecção
    processed_frame = preprocess_image(frame)
    
    # Detectar veículos
    vehicles = detect_vehicles(frame, frame_count)

    # Reconhecer placas
    recognize_plate(frame, frame_count)

    # Exibir frame processado
    cv2.imshow('Detections vehicles and plates', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
