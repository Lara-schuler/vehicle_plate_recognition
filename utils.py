import cv2
import os
import time

# Dicionário para armazenar placas recentemente detectadas com o tempo de última detecção
recent_plates = {}
save_interval_seconds = 30  # Intervalo mínimo entre salvamentos da mesma placa (em segundos)


def delete_old_files(directory, age_in_seconds):
    now = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > age_in_seconds:
                os.remove(file_path)
                print(f"Arquivo deletado: {file_path}")
                

def draw_stylized_plate_box(frame, x1, y1, x2, y2, text, rectangle_color=(0, 255, 0), 
                            text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5):
    # Cria a camada com transparência para o retângulo preenchido
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), rectangle_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Define o fundo do texto
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    text_bg_x2 = x1 + text_width + 10  # Ajusta a largura do fundo ao tamanho do texto
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (text_bg_x2, y1), bg_color, -1)

    # Adiciona o texto na imagem
    cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)


def draw_stylized_vehicle_box(frame, x, y, w, h, class_name):
    # Define as cores
    rectangle_color = (0, 255, 0)  # Verde para o retângulo
    text_color = (255, 255, 255)  # Branco para o texto
    text_bg_color = (0, 0, 0)  # Fundo preto para o texto

    # Desenhar a borda superior mais grossa
    thickness_top = 4  # Espessura da borda superior
    thickness_side = 4  # Espessura das bordas laterais
    cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, thickness_side)  # Bordas laterais
    cv2.line(frame, (x, y), (x + w, y), rectangle_color, thickness_top)  # Borda superior mais grossa

    # Adiciona o texto com fundo estilizado
    text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x + 5  # Pequeno espaço para a esquerda
    text_y = y - 10  # Acima do retângulo

    # Criar um fundo para o texto
    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0] + 10, text_y + 2), text_bg_color, -1)
    cv2.putText(frame, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    

def should_save_plate(detected_plate, recent_plates):
    """Verifica se a placa deve ser salva, com base no tempo da última detecção."""
    current_time = time.time()
    if detected_plate in recent_plates:
        last_seen_time = recent_plates[detected_plate]
        if current_time - last_seen_time < save_interval_seconds:
            return False
    recent_plates[detected_plate] = current_time
    return True

