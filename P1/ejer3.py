import cv2
import os
import numpy as np
def read_tif_file(file_path):
    # Read the .tif file
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # Return the processed image
    return image

# Path to the folder
folder_path = './Material_P1/Hierro'
pairs = []
# Read the paths of the files in the folder
file_paths = []
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        image = read_tif_file(os.path.join(folder_path, filename))
        # Extract the number from the image path
        image_number = (filename.split('_')[1].split('.')[0])
        image_number = image_number.replace("TE","")
        image_number = int(image_number)
        # Print the image number
        pairs.append((image,image_number))





imagen = pairs[0][0] #Para tomar las dimensiones
R2 = np.zeros(imagen.shape)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        tiempos = []
        #Valores del pixel i,j en las fotos
        y = list(map(lambda x: x[0][i,j], pairs))
        #Para el logaritmo quitamos los que son 0
        y = list(filter(lambda x: x != 0, y))
        #Cogemos los tiempos que tienen valores validos
        for (im,t) in pairs:
            if im[i,j] in y:
                tiempos.append(t)
        #Si hay mas de un punto
        if len(tiempos) > 1:
            #Calculamos la recta
            B,logA = np.polyfit(tiempos, np.log(y), 1,w=np.sqrt(y))
            #Con R2 se calculara el T2
            R2[i,j] = -B

R2_n = cv2.normalize(R2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

T2 = np.zeros(imagen.shape)
T2[R2_n != 0] =  1 / R2_n[R2_n != 0]
cv2.normalize(T2, T2, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

import matplotlib.pyplot as plt

# Create a figure with two subplots
fig, (ax2) = plt.subplots(1, 1)

# Plot R2
'''ax1.imshow(R2_n, cmap='gray')
ax1.set_title('R2 normalizado')
ax1.axis('off')'''
#ax1.colorbar()

# Plot T2
ax2.imshow(T2, cmap='gray')
ax2.set_title('T2 normalizado')
ax2.axis('off')
#ax2.colorbar()

# Display the figure
plt.show()

# Show T1 as a heatmap
plt.imshow(R2_n, cmap='hot')
plt.title('R2 Heatmap')
plt.axis('off')
plt.colorbar()

# Display the figure
plt.show()

# Create a figure with subplots
fig, axs = plt.subplots( 2,len(pairs)//2, figsize=(8, 8))

axs = axs.ravel()
# Iterate over the pairs and plot the images
for i, (image, image_number) in enumerate(pairs):
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(f'Image with TE {image_number}')
    axs[i].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()



