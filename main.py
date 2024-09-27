import cv2
from vehicle_detection import detect_vehicles
from plate_detection import detect_plate
from ocr_recognition import recognize_text
import os
import time

def save_image_in_folder(image, folder_name, image_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    cv2.imwrite(os.path.join(folder_name, image_name), image)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    processed_vehicle_ids = set()  # Inicializa um conjunto para IDs processados
    vehicle_count = 0  # Inicializa o contador de veículos
    start_time = time.time()  # Tempo de início
    i = 0  # Para nomear as imagens de forma única

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro ao capturar o frame.")
            break

        # Detecção de veículos
        vehicles = detect_vehicles(frame, processed_vehicle_ids)

        if len(vehicles) == 0:
            print("Nenhum veículo detectado")
        else:
            vehicle_count += len(vehicles)  # Contabiliza o número de veículos detectados
            print(f"Veículos detectados: {len(vehicles)} | Total acumulado: {vehicle_count}")

            for (x, y, w, h) in vehicles:
                vehicle_frame = frame[y:y + h, x:x + w]

                # Verificar se o frame do veículo não está vazio
                if vehicle_frame.size == 0:
                    print("Frame do veículo está vazio. Ignorando.")
                    continue

                # Salvar imagens de veículos em uma pasta específica
                save_image_in_folder(vehicle_frame, 'veiculos', f"vehicle_{i}.jpg")

                # Continuar com a detecção da placa
                plate, plate_coords = detect_plate(vehicle_frame)

                if plate is not None:
                    save_image_in_folder(plate, 'placas', f"plate_{i}.jpg")  # Salvar a imagem da placa
                    # Realizar OCR
                    plate_text = recognize_text(plate)
                    print(f"Texto da placa detectada: {plate_text}")

                i += 1  # Incrementa para nomear as próximas imagens de veículos

        # Verifica o tempo e imprime a contagem a cada 10 segundos
        if time.time() - start_time >= 10:
            print(f"Total de veículos no último intervalo: {vehicle_count}")
            vehicle_count = 0  # Reinicia o contador
            start_time = time.time()  # Reinicia o tempo

        # Exibir o vídeo com a detecção
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video('C:\\Users\\Lara Schüler\\Downloads\\carros.mp4')
