import cv2
import numpy as np


def to_grayscale(image):
    """Pasamos la imagen a blanco y negro[cite: 38]."""
    # Se ua la conversión estándar de BGR a gris[cite: 39].
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width, height):
    """Cambiamos el tamaño de la imagen[cite: 40]."""
    # La ajustamos a lo que nos pidan usando INTER_AREA[cite: 41].
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_blur(image, kernel_size=5):
    """Le ponemos un poco de desenfoque[cite: 42]."""
    # Usamos el filtro Gaussiano para quitarle el ruido[cite: 43].
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """Le movemos al brillo y al contraste[cite: 44]."""
    # Alpha es para el contraste y beta para el brillo[cite: 45].
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_threshold(image, thresh_value=127):
    """Binarizamos la imagen (solo blanco o negro)[cite: 49]."""
    if len(image.shape) != 2:
        raise ValueError("apply_threshold requiere una imagen en escala de grises (1 canal).")
    
    # Si pasa el umbral se vuelve blanco (255), si no, negro (0)[cite: 50].
    _, binarized = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return binarized


def detect_edges(image, low=50, high=150):
    """Buscamos los bordes de las figuras[cite: 52]."""
    # Si viene en color, primero hay que pasarla a gris[cite: 53].
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos Canny para resaltar los contornos[cite: 53].
    return cv2.Canny(image, low, high)


def full_pipeline(image, target_width=224, target_height=224):
    """Hacemos todo el proceso de un solo tiro[cite: 54]."""
    # Primero la redimensionamos[cite: 55].
    img = resize_image(image, target_width, target_height)
    
    # La pasamos a escala de grises[cite: 55].
    img = to_grayscale(img)
    
    # Le damos un suavizado leve (kernel=3)[cite: 55].
    img = apply_blur(img, kernel_size=3)
    
    # Y terminamos sacando los bordes con Canny[cite: 55].
    img = detect_edges(img, low=50, high=150)
    
    return img