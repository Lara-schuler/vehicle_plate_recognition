# plate_recognition.py

import cv2
from openalpr import Alpr
import re
import os
from utils import adjust_image_for_openalpr  # Nova importação para processamento de imagem

# Configurações do OpenALPR e diretórios
alpr = Alpr("br", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    raise Exception("Erro ao carregar OpenALPR")
alpr.set_top_n(10)
alpr.set_default_region("br")
conf_threshold = 82

placa_pattern = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")
output_dir_placas_raw = 'plates_raw'
output_dir_placas_processed = 'plates_processed'

# Criar diretórios de saída
os.makedirs(output_dir_placas_raw, exist_ok=True)
os.makedirs(output_dir_placas_processed, exist_ok=True)

# Função de reconhecimento de placas
def recognize_plate(frame, frame_count):
    original_img = frame.copy()

    # Pré-processar imagem para OpenALPR
    processed_img = adjust_image_for_openalpr(original_img)
    img_str = cv2.imencode('.jpg', processed_img)[1].tobytes()

    # Realizar reconhecimento com OpenALPR
    results = alpr.recognize_array(img_str)
    for plate in results['results']:
        detected_plate = plate['plate']
        confidence = plate['confidence']

        if placa_pattern.match(detected_plate) and confidence > conf_threshold:
            x1 = plate['coordinates'][0]['x']
            y1 = plate['coordinates'][0]['y']
            x2 = plate['coordinates'][2]['x']
            y2 = plate['coordinates'][2]['y']
            
            padding_x = 3  # Padding horizontal (esquerda/direita)
            padding_y = 1   # Padding vertical (acima/abaixo)

            x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
            x2, y2 = min(frame.shape[1], x2 + padding_x), min(frame.shape[0], y2 + padding_y)

            
            plate_img = original_img[y1:y2, x1:x2]

            if plate_img.size > 0:
                # Salvar imagem recortada sem tratamento
                cv2.imwrite(f'{output_dir_placas_raw}/plate_img_raw_{frame_count}.jpg', plate_img)

                # Processar e redimensionar placa antes da leitura
                processed_plate_img = adjust_image_for_openalpr(plate_img)

                # Salvar imagem processada
                cv2.imwrite(f'{output_dir_placas_processed}/plate_img_processed_{frame_count}.jpg', processed_plate_img)

                # Realizar leitura do OpenALPR na imagem processada
                processed_img_str = cv2.imencode('.jpg', processed_plate_img)[1].tobytes()
                adjusted_results = alpr.recognize_array(processed_img_str)

                if adjusted_results['results']:
                    adjusted_plate = adjusted_results['results'][0]['plate']
                    adjusted_confidence = adjusted_results['results'][0]['confidence']
                    print(f"Placa ajustada: {adjusted_plate} Confiança: {adjusted_confidence:.2f}")

            print(f"Placa detectada: {detected_plate} Confiança: {confidence:.2f}")
            cv2.putText(frame, f"Placa: {detected_plate} Conf: {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
