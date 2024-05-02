import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
def nifti_to_opencv(image_path, index):
    # Cargar la imagen desde el archivo NIfTI
    imagen_nifti = nib.load(image_path)
    
    # Obtener los datos de la imagen
    imagen = imagen_nifti.get_fdata()[:, :, index]
    
    # Normalizar los valores de píxeles entre 0 y 255 (opcional)
    imagen = (imagen - imagen.min()) / (imagen.max() - imagen.min()) * 255
    
    # Convertir la imagen a formato compatible con OpenCV (uint8)
    imagen_opencv = cv2.convertScaleAbs(imagen)
    
    return imagen_opencv

def aplicar_transformacion_rigida(imagen, angulo_rotacion, traslacion_x, traslacion_y):
    # Obtenemos las dimensiones de la imagen
    alto, ancho = imagen.shape[:2]
    
    # Calculamos el punto central de la imagen
    centro = (ancho // 2, alto // 2)
    
    # Definimos la matriz de transformación
    matriz_transformacion = cv2.getRotationMatrix2D(centro, angulo_rotacion, 1)
    
    # Aplicamos la traslación
    matriz_transformacion[0, 2] += traslacion_x
    matriz_transformacion[1, 2] += traslacion_y
    
    # Aplicamos la transformación a la imagen
    imagen_transformada = cv2.warpAffine(imagen, matriz_transformacion, (ancho, alto))
    
    return imagen_transformada

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar la imagen
    index = 93
    imagen = nifti_to_opencv("images/mr1.nii",index)
    #plt.imshow(imagen, cmap='gray')
    # Parámetros de la transformación
    angulo_rotacion = 4
    traslacion_x = 2
    traslacion_y = 5
    
    # Aplicar la transformación
    imagen_transformada = aplicar_transformacion_rigida(imagen, angulo_rotacion, traslacion_x, traslacion_y)
    
    

    from interpolation import trilinear_interpolation
    
    imagen_interpolada = trilinear_interpolation(imagen,imagen_transformada,0.1)

    from mutual_info import calcular_mutual_info

    mutual_info = calcular_mutual_info(imagen, imagen_interpolada)

    print("Métrica de Información Mutua entre imagen original e imagen interpolada:", mutual_info)
    # Mostrar la imagen original y la transformada
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen Transformada", imagen_transformada)
    cv2.imshow("Imagen Interpolada", imagen_interpolada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
