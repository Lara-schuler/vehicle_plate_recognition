import cv2

# Inicializar variáveis globais
previous_plate_coords = None
stable_frames = 0
STABLE_THRESHOLD = 5  # Número de quadros que a placa deve ser detectada

def detect_plate(vehicle_frame):
    gray = cv2.cvtColor(vehicle_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate = None
    plate_coords = None

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = w * h
            
            # Ajuste os valores conforme necessário
            if 0.4 < aspect_ratio < 5.0 and area > 1000:  # Ajuste os limites de área e proporção
                plate = vehicle_frame[y:y+h, x:x+w]
                plate_coords = (x, y, w, h)
                
                print(f"Placa detectada nas coordenadas: {plate_coords}")
                
                # Saia do loop após detectar a primeira placa válida
                return plate, plate_coords
    
    return None, None  # Se não encontrar nenhuma placa válida
