% Demo of "Chan Vese Level Sets"
%
% Example:
% seg_demo
%
% Coded by: Shawn Lankton (www.shawnlankton.com)

clc;
close all;
clear all;
I = imread('ec.jpg');  %-- load the image
m = zeros(size(I,1),size(I,2));          %-- create initial mask
m(110:150,110:150) = 1; 
%m(100:125,100:125) = 1; % Inicial



subplot(2,2,1); imshow(I); title('Input Image');
subplot(2,2,2); imshow(m); title('Initialization');
subplot(2,2,3); title('Segmentation');

% Inicial = 800, 3.0
 %seg = chanvese(I,init_mask,max_its,alpha, thershold, max_area,display)
seg = chanvese(I, m, 600, 1, 0.001, 11111); %-- Run segmentation
%seg = chanvese(I, m, 800, 3.0); %Inicial

subplot(2,2,4); imshow(seg); title('Global Region-Based Segmentation');


