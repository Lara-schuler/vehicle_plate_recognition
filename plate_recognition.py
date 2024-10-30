import cv2
from openalpr import Alpr
import re
import os
import time 

# Configurações do OpenALPR e diretórios
alpr = Alpr("br", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    raise Exception("Erro ao carregar OpenALPR")
alpr.set_top_n(10)
alpr.set_default_region("br")
conf_threshold = 50

placa_pattern = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

output_dir_placas_raw = 'plates_raw'
if not os.path.exists(output_dir_placas_raw):
    os.makedirs(output_dir_placas_raw)

# Função de reconhecimento de placas
def recognize_plate(frame, frame_count):
    original_img = frame.copy()
    # Captura o tempo de início
    start_time = time.time()
    img_str = cv2.imencode('.jpg', original_img)[1].tobytes()

    # Realizar reconhecimento com OpenALPR
    results = alpr.recognize_array(img_str)
    for plate in results['results']:
        if plate['matches_template']:
            
            detected_plate = plate['plate']
            confidence = plate['confidence']

            if placa_pattern.match(detected_plate) and confidence > conf_threshold:
                x1 = plate['coordinates'][0]['x']
                y1 = plate['coordinates'][0]['y']
                x2 = plate['coordinates'][2]['x']
                y2 = plate['coordinates'][2]['y']
            
                padding_x = 10  # Padding horizontal (esquerda/direita)
                padding_y = 10  # Padding vertical (acima/abaixo)

                x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                x2, y2 = min(frame.shape[1], x2 + padding_x), min(frame.shape[0], y2 + padding_y)

                # Recorte da placa detectada
                plate_img = original_img[y1:y2, x1:x2]

                if plate_img.size > 0:
                    # Salvar imagem recortada sem tratamento
                    cv2.imwrite(f'{output_dir_placas_raw}/plate_img_raw_{frame_count}.jpg', plate_img)

                # Exibir informações da placa detectada
                print(f"Placa detectada: {detected_plate} - Confiança: {confidence:.2f} - Tempo de processamento: {time.time() - start_time:.4f} segundos")
                cv2.putText(frame, f"Placa: {detected_plate}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            