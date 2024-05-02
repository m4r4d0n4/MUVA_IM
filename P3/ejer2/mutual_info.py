import numpy as np
from sklearn.metrics import mutual_info_score

def calcular_mutual_info(image1, image2):
    # Asegurarse de que las imágenes tengan el mismo tamaño
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño"

    # Normalizar los valores de píxeles a [0, 255]
    image1_norm = (image1 - image1.min()) / (image1.max() - image1.min()) * 255
    image2_norm = (image2 - image2.min()) / (image2.max() - image2.min()) * 255

    # Convertir los valores de píxeles a enteros
    image1_norm = image1_norm.astype(np.uint8)
    image2_norm = image2_norm.astype(np.uint8)

    # Calcular la métrica de información mutua
    mi = mutual_info_score(image1_norm.ravel(), image2_norm.ravel())

    return mi