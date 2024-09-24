import cv2
from vehicle_detection import detect_vehicles
from plate_detection import detect_plate
from ocr_recognition import recognize_text

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecção de veículos
        vehicles = detect_vehicles(frame)

        if len(vehicles) == 0:
            print("Nenhum veículo detectado")

        for (x, y, w, h) in vehicles:
            vehicle_frame = frame[y:y+h, x:x+w]

            # Detecção da placa
            plate, plate_coords = detect_plate(vehicle_frame)
            
            if plate is not None and plate_coords is not None:
                plate_x, plate_y, plate_w, plate_h = plate_coords
                
                # Desenhar retângulo ao redor da placa
                cv2.rectangle(frame, (x + plate_x, y + plate_y), (x + plate_x + plate_w, y + plate_y + plate_h), (255, 0, 0), 2)

                # OCR na placa
                plate_text = recognize_text(plate)
                print(f"Placa detectada: {plate_text}")
            else:
                print("Placa não reconhecida")

            # Saia do loop após processar o primeiro veículo (para testar)
            break

        # Exibir o vídeo com a detecção
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video('C:\\Users\\Lara Schüler\\Downloads\\carros.mp4')
