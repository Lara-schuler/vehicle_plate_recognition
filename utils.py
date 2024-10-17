import cv2
import os
import time

def preprocess_image(plate_image):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Equalization of the histogram to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Bilateral filter for noise reduction while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def delete_old_files(directory, age_in_seconds):
    now = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > age_in_seconds:
                os.remove(file_path)
                print(f"Arquivo deletado: {file_path}")
                
                

def resize_plate_image(plate_image, target_width=300):
    height, width = plate_image.shape[:2]
    scaling_factor = target_width / width
    resized = cv2.resize(plate_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    return resized
