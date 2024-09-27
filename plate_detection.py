import cv2
import time  # Importar a biblioteca time

# Inicializar variáveis globais
previous_plate_coords = None
stable_frames = 0
STABLE_THRESHOLD = 2  # Número de quadros que a placa deve ser detectada

def detect_plate(vehicle_frame):
    # Verificar se o frame do veículo é válido
    if vehicle_frame is None or vehicle_frame.size == 0:
        print("Frame do veículo está vazio. Ignorando.")
        return None, None

    start_time = time.time()  # Captura o tempo de início

    # Converter o frame do veículo para escala de cinza
    gray = cv2.cvtColor(vehicle_frame, cv2.COLOR_BGR2GRAY)

    # Equalizar o histograma para melhorar o contraste da imagem
    gray = cv2.equalizeHist(gray)

    # Aumentar o contraste e o brilho para melhorar a detecção de bordas
    enhanced_gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate = None
    plate_coords = None

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Verifica se o contorno é um retângulo
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = w * h
            
            # Ajustar os valores conforme necessário
            if 2.0 < aspect_ratio < 5.0 and area > 1500:  # Ajustes de proporção e área
                plate = vehicle_frame[y:y + h, x:x + w]
                plate_coords = (x, y, w, h)

                # Desenhar o retângulo cinza ao redor da placa detectada
                cv2.rectangle(vehicle_frame, (x, y), (x + w, y + h), (128, 128, 128), 2)  # Retângulo cinza
                print(f"Placa detectada nas coordenadas: {plate_coords}")

                # Calcular o tempo decorrido
                elapsed_time = time.time() - start_time
                print(f"Tempo para detectar a placa: {elapsed_time:.4f} segundos")

                # Retorna a primeira placa válida
                return plate, plate_coords

    return None, None  # Se não encontrar nenhuma placa válida
