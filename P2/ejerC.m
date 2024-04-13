% Leer la imagen 
imagen = imread('higado.bmp');

% Imagen en escala de grises
imagen_gris = rgb2gray(imagen);

% Paso 1: Calcular la imagen de gradiente
mascara = fspecial('sobel'); % Crear la mascara de Sobel
gradiente_x = imfilter(imagen_gris, mascara'); % Calcular el gradiente en dirección x
gradiente_y = imfilter(imagen_gris, mascara); % Calcular el gradiente en dirección y
gradiente = sqrt(double(gradiente_x).^2 + double(gradiente_y).^2); % Calcular el modulo de los gradientes

% Paso 2: Aplicar Watershed a la imagen de gradiente
transformada_distancia = bwdist(~imbinarize(gradiente)); % Transformada de distancia
transformada_distancia = -transformada_distancia;
transformada_distancia(~imbinarize(gradiente)) = -Inf; % Establecer los minimos locales
segmentacion = watershed(transformada_distancia); % Aplicar Watershed
segmentacion(~mascara) = 0;

% Paso 3: Utilizar imimposemin para imponer mínimos en las zonas adecuadas
% Operaciones de morfologia matematica para limpiar la imagen y mejorar la segmentacion
I = imagen_gris;
se = strel("disk",20);
Io = imopen(I,se);
Ie = imerode(I,se);
Iobr = imreconstruct(Ie,I);
Ioc = imclose(Io,se);
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);

% Identificar maximos regionales
fgm = imregionalmax(Iobrcbr);
se2 = strel(ones(5,5));
fgm2 = imclose(fgm,se2);
fgm3 = imerode(fgm2,se2);
fgm4 = bwareaopen(fgm3,20);

% Binarizar la imagen de fondo
bw = imbinarize(Iobrcbr);
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;

% Imponer minimos locales en las regiones adecuadas
minimos = imimposemin(gradiente, bgm | fgm4);

% Paso 4: Obtener el resultado de Watershed sobre la imagen modificada
segmentacion_final = watershed(minimos);

% Visualizacion de los resultados
figure;
subplot(2,2,1); imshow(imagen); title('Imagen Original');
subplot(2,2,2); imshow(gradiente); title('Imagen de Gradiente');
subplot(2,2,3); imshow(label2rgb(segmentacion,'jet',[.5 .5 .5])); title('Resultado Watershed');
subplot(2,2,4); imshow(label2rgb(segmentacion_final)); title('Resultado con mínimos impuestos');
