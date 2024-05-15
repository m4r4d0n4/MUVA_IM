%% Práctica 2. Apartado B: Segmentación usando EM
% Segmentación de imágenes a partir de una estimación
% de máxima verosimilitud de los parámetros del modelo
% estadístico.
% Parámetros de un modelo de mezcla de distribuciones gaussianas

close all;
clear all;
clc;
image = imread("brain.bmp");
gray = rgb2gray(image);

% Segmentación con 4 clases
[mask_4,mu_4,v_4,p_4]=EMSeg(gray,4);

% Segmentación con 6 clases (Volumen)
classes = 6;
[mask_6, mu_6, v_6, p_6]= EMSeg(gray, classes);
time_6 = toc;

% Display the original image and the segmentation mask
figure(1);
subplot(2, 1, 1);
imshow(image);
title('Original Image');
subplot(2, 2, 3);
imshow(double(image).*mask_4);
colormap('jet');
title('Segmentation Mask 4');
subplot(2,2,4);
imshow(double(image).*mask_6);
colormap('jet');
title('Segmentation Mask 6');


%%%%%%
%% B) Estudio de la segmentación con inicialización
classes = 6;
[mask0, mu, v, p] = EMSeg(gray, classes);
[mask1, ~, ~, ~] = EMSeg(gray, classes, mu, v, p);

figure(2);
subplot(2, 2, 1);
imshow(image);
title('Imagen original')
subplot(2, 2, 2);
imshow(double(image).*mask0);
title('Segmentación sin inicializar')
subplot(2,2,3);
imshow(double(image).*mask1);
title('Segmentación con inicialización')
subplot(2,2,4);
imshowpair(mask0,mask1,'diff');
title('Diferencia entre segmentaciones')
sgtitle('Apartado P2/2.B') 




