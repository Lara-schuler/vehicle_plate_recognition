import cv2
import numpy as np
import os
import time

def adjust_image_for_openalpr(plate_image):
    # Conversão para escala de cinza
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Equalização do histograma para melhorar o contraste
    enhanced = cv2.equalizeHist(gray)
    
    # Aplicar desfoque Gaussiano leve para preservar bordas e reduzir ruído
    filtered = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Verificação do redimensionamento
    height, width = filtered.shape[:2]
    if height < 50 or width < 120:  # Critério para redimensionamento
        scale_factor = 2
        filtered = cv2.resize(filtered, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)
    
    return filtered




def delete_old_files(directory, age_in_seconds):
    now = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > age_in_seconds:
                os.remove(file_path)
                print(f"Arquivo deletado: {file_path}")
                



