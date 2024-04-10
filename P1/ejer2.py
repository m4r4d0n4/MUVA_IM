import cv2
import numpy as np
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
import dif_aniso as da
def nonLocalMeans(noisy, params = tuple(), verbose = True): ### REFERENCIA : https://github.com/praveenVnktsh/Non-Local-Means/blob/main/main.py
  '''
  Performs the non-local-means algorithm given a noisy image.
  params is a tuple with:
  params = (bigWindowSize, smallWindowSize, h)
  Please keep bigWindowSize and smallWindowSize as even numbers
  '''

  bigWindowSize, smallWindowSize, h  = params
  padwidth = bigWindowSize//2
  image = noisy.copy()

  # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
  paddedImage = np.zeros((image.shape[0] + bigWindowSize,image.shape[1] + bigWindowSize))
  paddedImage = paddedImage.astype(np.uint8)
  paddedImage[padwidth:padwidth+image.shape[0], padwidth:padwidth+image.shape[1]] = image
  paddedImage[padwidth:padwidth+image.shape[0], 0:padwidth] = np.fliplr(image[:,0:padwidth])
  paddedImage[padwidth:padwidth+image.shape[0], image.shape[1]+padwidth:image.shape[1]+2*padwidth] = np.fliplr(image[:,image.shape[1]-padwidth:image.shape[1]])
  paddedImage[0:padwidth,:] = np.flipud(paddedImage[padwidth:2*padwidth,:])
  paddedImage[padwidth+image.shape[0]:2*padwidth+image.shape[0], :] =np.flipud(paddedImage[paddedImage.shape[0] - 2*padwidth:paddedImage.shape[0] - padwidth,:])
  


  iterator = 0
  totalIterations = image.shape[1]*image.shape[0]*(bigWindowSize - smallWindowSize)**2

  if verbose:
    print("TOTAL ITERATIONS = ", totalIterations)

  outputImage = paddedImage.copy()

  smallhalfwidth = smallWindowSize//2


  # For each pixel in the actual image, find a area around the pixel that needs to be compared
  for imageX in range(padwidth, padwidth + image.shape[1]):
    for imageY in range(padwidth, padwidth + image.shape[0]):
      
      bWinX = imageX - padwidth
      bWinY = imageY - padwidth

      #comparison neighbourhood
      compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth + 1,imageX-smallhalfwidth:imageX+smallhalfwidth + 1]
      
      
      pixelColor = 0
      totalWeight = 0

      # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
      for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
        for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):   
          #find the small box       
          smallNbhd = paddedImage[sWinY:sWinY+smallWindowSize + 1,sWinX:sWinX+smallWindowSize + 1]
          euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
          #weight is computed as a weighted softmax over the euclidean distances
          weight = np.exp(-euclideanDistance/h)
          totalWeight += weight
          pixelColor += weight*paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]
          iterator += 1

          if verbose:
            percentComplete = iterator*100/totalIterations
            if percentComplete % 5 == 0:
              print('% COMPLETE = ', percentComplete)

      pixelColor /= totalWeight
      outputImage[imageY, imageX] = pixelColor

  return outputImage[padwidth:padwidth+image.shape[0],padwidth:padwidth+image.shape[1]]


def add_gaussian_noise(image, mean=0, std_dev=0.5):
    """
    Agrega ruido gaussiano a una imagen.
    """
    noisy_image = np.copy(image)
    h, w = image.shape
    noise = np.random.normal(mean, std_dev, (h, w)).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


# Cargar la imagen
original_image = cv2.imread('Material_P1/T1.png', cv2.IMREAD_GRAYSCALE)

# Agregar ruido gaussiano a la imagen
noisy_image = add_gaussian_noise(original_image)

gParams = {
    'bigWindow' : 20,
    'smallWindow':6,
    'h':14,
}
imagen_filtrogaussiano = cv2.GaussianBlur(noisy_image, (5, 5), 0)
anisotropic_img = da.PeronaMalik_Smoother(noisy_image, 5, 0.01, "Exponential",15,False)[-1]
#perform NLM filtering
filtered_image = nonLocalMeans(noisy_image, params = (gParams['bigWindow'], gParams['smallWindow'],gParams['h']))

# Mostrar las imágenes
plt.figure(figsize=(15, 5))
#
plt.subplot(1, 5, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Imagen con Ruido Gaussiano')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Imagen Filtrada con NLM')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(anisotropic_img, cmap='gray')
plt.title('Imagen Filtrada con Perona Malik')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(imagen_filtrogaussiano, cmap='gray')
plt.title('Imagen Filtrada con Filtro Gaussiano')
plt.axis('off')




plt.show()

