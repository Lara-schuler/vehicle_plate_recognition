import cv2
import pytesseract

# Configurações do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carregar a imagem
image_path = 'images/plate1.jpg'
image = cv2.imread(image_path)

# Função para cortar a região da placa
def crop_plate_region(image):
    height, width, _ = image.shape
    x_start = int(width * 0.11)
    x_end = int(width * 0.92)
    y_start = int(height * 0.33)
    y_end = int(height * 0.85)
    return image[y_start:y_end, x_start:x_end]

# Função de pré-processamento da imagem
def preprocess_image(image):
    image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Tentar usar uma limiarização binária simples
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Estrutura de dilatação e erosão
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    return thresh

# Função para ordenar os contornos com base na posição horizontal
def sort_contours(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, _ = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
    return contours

# Função de segmentação dos caracteres
def segment_characters(thresh, margin=5):
    # Encontre contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Ordenar os contornos da esquerda para a direita (horizontalmente)
    contours = sort_contours(contours)
    
    # Inverter a ordem dos contornos, se necessário
    contours = contours[::-1]  # Inverte a lista de contornos

    # Criar uma lista para armazenar os caracteres
    characters = []
    
    for contour in contours:
        # Obter retângulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adicionar margem à caixa delimitadora
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(thresh.shape[1], x + w + margin)
        y_end = min(thresh.shape[0], y + h + margin)
        
        # Filtrar com base nas dimensões do caractere
        if w > 10 and h > 20:  # Ajuste esses valores conforme necessário
            character = thresh[y_start:y_end, x_start:x_end]
            characters.append(character)
            # Desenhar retângulos na imagem original para visualização
            cv2.rectangle(thresh, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    return characters, thresh

# Processar a imagem
cropped_image = crop_plate_region(image)
preprocessed_image = preprocess_image(cropped_image)

# Segmentar e ordenar os caracteres
segmented_characters, thresh_image_with_rectangles = segment_characters(preprocessed_image)

# Exibir a imagem segmentada com os contornos
cv2.imshow("Imagem Pre-processada com Contornos", thresh_image_with_rectangles)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Usar o Tesseract para ler a placa (testar com cada caractere)
config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 10 --oem 1'

# Processar cada caractere com o Tesseract
for i, character in enumerate(segmented_characters):
    if character.shape[0] > 0 and character.shape[1] > 0:
        resized_character = cv2.resize(character, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        plate_text = pytesseract.image_to_string(resized_character, lang='eng', config=config)
        print(f"Texto do caractere {i}: {plate_text.strip()}")

cv2.destroyAllWindows()
