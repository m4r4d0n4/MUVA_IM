import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def trilinear_interpolation(image1, image2, weights):
    # Verificar que las imágenes tengan las mismas dimensiones
    assert image1.shape == image2.shape, "Las imágenes deben tener las mismas dimensiones"

    # Crear la función interpoladora para la primera imagen
    interpolator_image1 = RegularGridInterpolator((np.arange(image1.shape[0]), np.arange(image1.shape[1])), image1)
    
    # Crear la función interpoladora para la segunda imagen
    interpolator_image2 = RegularGridInterpolator((np.arange(image2.shape[0]), np.arange(image2.shape[1])), image2)

    # Calcular la interpolación trilineal para cada punto en el espacio bidimensional
    interpolated_values = np.zeros_like(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            position = np.array([i, j])
            value1 = interpolator_image1(position)
            value2 = interpolator_image2(position)
            interpolated_values[i, j] = value1 * (1 - weights) + value2 * weights
    
    return interpolated_values

