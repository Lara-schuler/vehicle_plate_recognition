import cv2

def detect_vehicles(frame):
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar veículos
    vehicles = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    print(f"Veículos detectados: {len(vehicles)}")  # Para monitorar quantos veículos são detectados
    
    # Filtrar veículos muito pequenos ou grandes
    filtered_vehicles = []
    for (x, y, w, h) in vehicles:
        if 60 < w < 400 and 60 < h < 400:  # Limites de tamanho para o veículo
            filtered_vehicles.append((x, y, w, h))

    return filtered_vehicles
