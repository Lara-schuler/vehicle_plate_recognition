import cv2
from vehicle_detection import detect_vehicles
from plate_recognition import recognize_plate
from utils import adjust_image_for_openalpr, delete_old_files  # Importação corrigida

# Configurações
video_path = 'C:\\Users\\Lara Schüler\\Downloads\\transito.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Configurações de tempo para deletar arquivos antigos
age_in_seconds = 2 * 60  # 2 minutos
# age_in_seconds = 48 * 60 * 60  # 48 horas
output_dir_placas_raw = 'plates_raw'
output_dir_placas_processed = 'plates_processed'
output_dir_veiculos = 'vehicles'
#output_dir_placas_roi = 'plates_roi'

# Chamar a função para deletar arquivos antigos
delete_old_files(output_dir_placas_raw, age_in_seconds)
delete_old_files(output_dir_placas_processed, age_in_seconds)
delete_old_files(output_dir_veiculos, age_in_seconds)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou falha ao carregar frame")
        break

    # Redimensionar o frame para um tamanho menor (facilita o processamento)
    new_width = 800
    new_height = int(frame.shape[0] * (new_width / frame.shape[1]))
    frame = cv2.resize(frame, (new_width, new_height))

    # Detectar veículos
    vehicles = detect_vehicles(frame, frame_count)

    # Reconhecer e ajustar placas
    recognize_plate(frame, frame_count)

    # Exibir o frame processado com as detecções
    cv2.imshow('Detections vehicles and plates', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
