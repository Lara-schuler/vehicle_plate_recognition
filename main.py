import cv2
from vehicle_detection import detect_vehicles
from plate_recognition import recognize_plate
from utils import preprocess_image

# Configurações
video_path = 'C:\\Users\\Lara Schüler\\Downloads\\carros.mp4'
cap = cv2.VideoCapture(video_path)
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
    cv2.imshow('Veículos e Placas Detectados', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
