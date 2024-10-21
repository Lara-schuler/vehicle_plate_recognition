import cv2
import os
import time

def preprocess_image(plate_image):
    # Converter para escala de cinza
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Reduzir o efeito do filtro bilateral
    gray = cv2.bilateralFilter(gray, 5, 15, 15)  # Ajuste os parâmetros aqui
    
    # Tente reduzir o desfoque
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Experimente (3, 3) ou (5, 5)

    # Limiarização fixa em vez de adaptativa
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)  # Ajuste o valor de 127 se necessário

    # Dilatação leve
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)  # Ajuste o número de iterações se necessário

    # Aplique um filtro de nitidez
    sharpened = cv2.addWeighted(dilated, 1.5, dilated, -0.5, 0)

    return sharpened  # Retorne a imagem processada


def delete_old_files(directory, age_in_seconds):
    now = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > age_in_seconds:
                os.remove(file_path)
                print(f"Arquivo deletado: {file_path}")
                
                

# utils.py
def resize_plate_image(plate_image, scale_factor=5):
    # Aumentar o tamanho da imagem da placa em 4x
    height, width = plate_image.shape[:2]
    resized = cv2.resize(plate_image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    return resized




