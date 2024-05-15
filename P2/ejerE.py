import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
# Leer las imágenes
mr_image = sitk.ReadImage("MaterialP2/MR.nii")
ft1_image = sitk.ReadImage("MaterialP2/fT1.nii")
flabels_image = sitk.ReadImage("MaterialP2/fLabels.nii")

# Convertir las imágenes a arrays numpy
mr_array = sitk.GetArrayFromImage(mr_image)
#print(mr_array.shape)
ft1_array = sitk.GetArrayFromImage(ft1_image)
flabels_array = sitk.GetArrayFromImage(flabels_image)

# Crear un subplot con 1 fila y 3 columnas
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Las imágenes volumétricas
images = [mr_array, ft1_array, flabels_array]
titles = ['MR', 'fT1', 'fLabels']
index = 15
# Mostrar las primeras tres imágenes en un solo plot
for i in range(3):
    axes[i].imshow(images[i][index, :, :], cmap='gray')  # Seleccionar la imagen correspondiente al elemento 15
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.show()

# Definir las coordenadas de la ROI en mr_array (ejemplo)
x_start_roi, y_start_roi, z_start_roi = 20, 50, 50
x_end_roi, y_end_roi, z_end_roi = 30, 150, 150

# Crear una matriz vacía para almacenar los resultados en la ROI
resultados_roi = np.empty((x_end_roi - x_start_roi, y_end_roi - y_start_roi, z_end_roi - z_start_roi))

# Iterar sobre cada voxel dentro de la ROI
for x in range(x_start_roi, x_end_roi):
    for y in range(y_start_roi, y_end_roi):
        for z in range(z_start_roi, z_end_roi):
            # Coordenadas del voxel de referencia en la imagen derecha
            x_ref, y_ref, z_ref = x, y, z
            
            # Definir las coordenadas de la región 11x11x11 en la imagen derecha
            x_start = max(x_ref - 5, x_start_roi)
            x_end = min(x_ref + 6, x_end_roi)
            y_start = max(y_ref - 5, y_start_roi)
            y_end = min(y_ref + 6, y_end_roi)
            z_start = max(z_ref - 5, z_start_roi)
            z_end = min(z_ref + 6, z_end_roi)

            # Inicializar el voxel más similar y su diferencia mínima
            voxel_mas_similar = None
            diferencia_minima = float('inf')

            # Iterar sobre la región 11x11x11 en la imagen derecha
            for xi in range(x_start, x_end):
                for yi in range(y_start, y_end):
                    for zi in range(z_start, z_end):
                        # Definir las coordenadas del kernel 3x3x3 alrededor del voxel de referencia
                        x_kernel_start = max(xi - 1, x_start)
                        x_kernel_end = min(xi + 2, x_end)
                        y_kernel_start = max(yi - 1, y_start)
                        y_kernel_end = min(yi + 2, y_end)
                        z_kernel_start = max(zi - 1, z_start)
                        z_kernel_end = min(zi + 2, z_end)

                        # Calcular la diferencia media en el kernel 3x3x3
                        diferencia_media = np.mean(np.abs(mr_array[x:x+1, y:y+1, z:z+1] - mr_array[x_kernel_start:x_kernel_end, y_kernel_start:y_kernel_end, z_kernel_start:z_kernel_end]))
                        
                        # Actualizar el voxel más similar si encontramos una diferencia mínima
                        if diferencia_media < diferencia_minima:
                            diferencia_minima = diferencia_media
                            voxel_mas_similar = flabels_array[xi, yi, zi]  # Voxel más similar encontrado

            # Almacenar el voxel más similar en los resultados
            resultados_roi[x - x_start_roi, y - y_start_roi, z - z_start_roi] = voxel_mas_similar



# Elementos a mostrar
elementos_a_mostrar = [0, 5, 9]

# Mostrar los elementos seleccionados
for idx, elemento in enumerate(elementos_a_mostrar, 1):
    plt.subplot(1, len(elementos_a_mostrar), idx)
    plt.imshow(resultados_roi[elemento, :, :], cmap='gray')  # Seleccionar el plano específico
    plt.colorbar()
    plt.title(f"Elemento {elemento}")
    plt.axis('off')

plt.show()