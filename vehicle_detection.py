import cv2
import numpy as np

def detect_vehicles(frame, processed_vehicle_ids):
    # Carregar a rede YOLO
    net = cv2.dnn.readNetFromDarknet('yolo/yolov4-tiny.cfg', 'yolo/yolov4-tiny.weights')

    # Carregar os nomes das classes
    with open('yolo/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Configurar a entrada para a rede
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Obter nomes das camadas
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Realizar a detecção
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    vehicle_classes = [2, 3, 5, 7]  # IDs que correspondem a veículos

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id in vehicle_classes:
                # Extraindo a posição e o tamanho da detecção
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Calculando as coordenadas da caixa
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box

            # Desenhar o retângulo ao redor do veículo detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Obter o nome da classe
            class_name = classes[class_ids[i]]
            # Adicionar texto acima do retângulo
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            detected_boxes.append([x, y, w, h])

    return detected_boxes
