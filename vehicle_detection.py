import cv2
import numpy as np
import os
from utils import draw_stylized_vehicle_box

# Carregar a rede YOLO
net = cv2.dnn.readNetFromDarknet('yolo/yolov4-tiny.cfg', 'yolo/yolov4-tiny.weights')
with open('yolo/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_classes = [2, 3, 5, 7]  # IDs que correspondem a veículos

output_dir_veiculos = 'vehicles'
if not os.path.exists(output_dir_veiculos):
    os.makedirs(output_dir_veiculos)
    
def detect_vehicles(frame, frame_count):
    # YOLO: Detecção de veículos
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7 and class_id in vehicle_classes:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            
             # Obter o nome da classe (veículo) antes da função draw_stylized_vehicle_box
            class_name = classes[class_ids[i]]

            # Desenha o retângulo e o texto estilizado ao redor do veículo detectado
            draw_stylized_vehicle_box(frame, x, y, w, h, class_name)

            # Recortar e salvar a imagem do veículo
            vehicle_img = frame[y:y+h, x:x+w]
            if vehicle_img.size > 0:
                cv2.imwrite(f'{output_dir_veiculos}/vehicle_img_{frame_count}_{i}.jpg', vehicle_img)

    