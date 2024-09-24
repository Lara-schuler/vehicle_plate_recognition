import cv2
import easyocr

# Inicializar o OCR uma vez
reader = easyocr.Reader(['pt'])  # Carregar o modelo para português (ou inglês 'en')

def recognize_text(plate_image):
    # Verificar se a imagem da placa é válida
    if plate_image is None:
        return "Imagem da placa inválida"

    # Verificar se a imagem está colorida e converter para escala de cinza, se necessário
    if len(plate_image.shape) == 3:  # Se a imagem tiver 3 canais (BGR)
        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_plate = plate_image  # Se já estiver em grayscale

    # Usar o EasyOCR para ler o texto da placa
    result = reader.readtext(gray_plate)

    print(f"Resultado do OCR: {result}")  # Adicionando log para ver o que está sendo detectado

    # Verificar se algum texto foi detectado
    if result:
        return result[0][1]  # Retornar o texto da primeira detecção (com maior confiança)
    else:
        return "Placa não reconhecida"
