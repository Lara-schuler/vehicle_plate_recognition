import cv2
from openalpr import Alpr
import re
import os
import pytesseract
from utils import resize_plate_image, preprocess_image

# Inicializar OpenALPR
alpr = Alpr("br", "openalpr.conf", "runtime_data")
if not alpr.is_loaded():
    raise Exception("Erro ao carregar OpenALPR")
alpr.set_top_n(10)
alpr.set_default_region("br")
conf_threshold = 60

placa_pattern = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

output_dir_placas_raw = 'plates_raw'  # Pasta para placas recortadas sem tratamento
output_dir_placas_processed = 'plates_processed'  # Pasta para placas tratadas

# Criação das pastas se não existirem
if not os.path.exists(output_dir_placas_raw):
    os.makedirs(output_dir_placas_raw)
if not os.path.exists(output_dir_placas_processed):
    os.makedirs(output_dir_placas_processed)


# Função para reconhecer e processar a placa
def recognize_plate(frame, frame_count):
    original_img = frame.copy()

    # Converter frame para escala de cinza para melhor detecção pelo OpenALPR
    processed_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_str = cv2.imencode('.jpg', processed_img)[1].tobytes()

    results = alpr.recognize_array(img_str)
    for plate in results['results']:
        detected_plate = plate['plate']
        confidence = plate['confidence']

        # Verificar se a placa tem alta confiança e está no padrão esperado
        if placa_pattern.match(detected_plate) and confidence > conf_threshold:
            x1 = plate['coordinates'][0]['x']
            y1 = plate['coordinates'][0]['y']
            x2 = plate['coordinates'][2]['x']
            y2 = plate['coordinates'][2]['y']
            
            padding = 10
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            
            plate_img = original_img[y1:y2, x1:x2]

            # Verificar se a imagem da placa não está vazia
            if plate_img.size > 0:
                # Salvar a imagem da placa recortada sem tratamento
                cv2.imwrite(f'{output_dir_placas_raw}/plate_img_raw_{frame_count}.jpg', plate_img)

                # Redimensionar a imagem da placa
                resized_plate_img = resize_plate_image(plate_img)  # Aqui a placa será redimensionada 4x

                # Aplicar pré-processamento na imagem redimensionada
                preprocessed_plate_img = new_func(resized_plate_img)

                # Salvar a imagem tratada da placa
                cv2.imwrite(f'{output_dir_placas_processed}/plate_img_processed_{frame_count}.jpg', preprocessed_plate_img)
                
                # Ler os caracteres da placa com o Tesseract (OCR)
                config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7 --oem 1'
                plate_text = pytesseract.image_to_string(preprocessed_plate_img, lang='eng', config=config)
                
                print(f"Texto da placa lido pelo Tesseract: {plate_text.strip()}")
            else:
                print(f"Imagem da placa vazia para o frame {frame_count}")

            # Exibir a placa detectada e sua confiança no frame
            cv2.putText(frame, f"Placa: {detected_plate} Conf: {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            print(f"Placa detectada: {detected_plate} Confiança: {confidence:.2f}")

def new_func(resized_plate_img):
    preprocessed_plate_img = preprocess_image(resized_plate_img)
    return preprocessed_plate_img
