import cv2
from vehicle_detection import detect_vehicles
from plate_recognition import recognize_plate
from utils import preprocess_image, delete_old_files

# Caminho da imagem a ser processada
image_path = 'images\\carro9.jpeg'

# Carregar a imagem
image = cv2.imread(image_path)
if image is None:
    print("Erro ao carregar a imagem.")
    exit()

frame_count = 0

# Configurações para deletar arquivos antigos
age_in_seconds = 10 * 60 # 10 minutos
output_dir_placas = 'plates_processed'
output_dir_veiculos = 'vehicles'

# Chamar a função para deletar arquivos antigos
delete_old_files(output_dir_placas, age_in_seconds)
delete_old_files(output_dir_veiculos, age_in_seconds)

# Definir os limites máximos de largura e altura
max_width = 800
max_height = 600

# Obter as dimensões atuais da imagem
height, width = image.shape[:2]

# Calcular a proporção de redimensionamento, mantendo a proporção original
scaling_factor = min(max_width / width, max_height / height)

# Redimensionar a imagem com base no fator de escala
new_width = int(width * scaling_factor)
new_height = int(height * scaling_factor)

# Redimensionar a imagem para caber nos limites definidos
image = cv2.resize(image, (new_width, new_height))

'''# Redimensionar a imagem (opcional, ajuste conforme necessário)
new_width = 800
new_height = int(image.shape[0] * (new_width / image.shape[1]))
image = cv2.resize(image, (new_width, new_height))'''

# Pré-processar a imagem antes da detecção
processed_image = preprocess_image(image)

# Detectar veículos
vehicles = detect_vehicles(image, frame_count)

# Reconhecer placas
recognize_plate(image, frame_count)

# Exibir a imagem processada
cv2.imshow('Detections vehicles and plates', image)

# Aguarda uma tecla para fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
