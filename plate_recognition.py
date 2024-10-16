import cv2
from openalpr import Alpr
import re
import os

# Inicializar OpenALPR
alpr = Alpr("br", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    raise Exception("Erro ao carregar OpenALPR")
alpr.set_top_n(10)
alpr.set_default_region("br")
conf_threshold = 80

placa_pattern = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

output_dir_placas = 'plates'
if not os.path.exists(output_dir_placas):
    os.makedirs(output_dir_placas)
    
print(f"Verificando diretório: {output_dir_placas}")

def recognize_plate(frame, frame_count):
    original_img = frame.copy()
    processed_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_str = cv2.imencode('.jpg', processed_img)[1].tobytes()

    results = alpr.recognize_array(img_str)
    for plate in results['results']:
        detected_plate = plate['plate']
        confidence = plate['confidence']

        if placa_pattern.match(detected_plate) and confidence > conf_threshold:
            x1 = plate['coordinates'][0]['x']
            y1 = plate['coordinates'][0]['y']
            x2 = plate['coordinates'][2]['x']
            y2 = plate['coordinates'][2]['y']
            
            padding = 10
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            
            plate_img = original_img[y1:y2, x1:x2]
            
            # Verificar se a imagem não está vazia
            if plate_img.size > 0:
                cv2.imwrite(f'{output_dir_placas}/plate_img_{frame_count}.jpg', plate_img)
            else:
                print(f"Imagem da placa vazia para o frame {frame_count}")

            cv2.putText(frame, f"Placa: {detected_plate} Conf: {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            print(f"Placa detectada: {detected_plate} Confiança: {confidence:.2f}")
