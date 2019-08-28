clear all;
clc;
close all;

imageNo = 2019;
load(strcat('../Data/',num2str(imageNo),'.mat'));

I = uint8(255 * mat2gray(cjdata.image));
angle = 290;
I = imrotate(I, angle);
cjdata.tumorMask = imrotate(cjdata.tumorMask, angle);

%Skull Stripping
% Get the dimensions of the image.
% numberOfColorBands should be = 1.
[rows, columns, numberOfColorBands] = size(I);

% Display the original gray scale image.
subplot(2, 3, 1);
imshow(I, []);
axis on;
title('Original Grayscale Image');
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Give a name to the title bar.
set(gcf, 'Name', 'Skull Stripping', 'NumberTitle', 'Off')
% Let's compute and display the histogram.
[pixelCount, grayLevels] = imhist(I);
subplot(2, 3, 2);
bar(grayLevels, pixelCount);
grid on;
title('Histogram of original image');
xlim([0 grayLevels(end)]); % Scale x axis manually.
% Crop image to get rid of light box surrounding the image
I = I(3:end-3, 4:end-4);
% Threshold to create a binary image
binaryImage = I > 20;
% Get rid of small specks of noise
binaryImage = bwareaopen(binaryImage, 10);
% Display the original gray scale image.
subplot(2, 3, 3);
imshow(binaryImage, []);
axis on;
title('Binary Image');
% Seal off the bottom of the head - make the last row white.
binaryImage(end,:) = true;
% Fill the image
binaryImage = imfill(binaryImage, 'holes');
subplot(2, 3, 4);
imshow(binaryImage, []);
axis on;
title('Cleaned Binary Image');
% Erode away 15 layers of pixels.
se = strel('disk', 15, 0);
binaryImage = imerode(binaryImage, se);
subplot(2, 3, 5);
imshow(binaryImage, []);
axis on;
title('Eroded Binary Image');
% Mask the gray image
finalImage = I; % Initialize.
finalImage(~binaryImage) = 0;
subplot(2, 3, 6);
imshow(finalImage, []);
axis on;
title('Skull stripped Image');

%Image Segmentation using Fuzzy C-Means
data=im2double(reshape(finalImage,[],1));
[center,member]=fcm(data,3);
[center,cidx]=sort(center);
member=member';
member=member(:,cidx);
[maxmember,label]=max(member,[],2);
level=(max(data(label==2))+min(data(label==3)))/2;
bw=im2bw(finalImage,level);

%se = strel('disk', 20);
%img_open = imopen(bw, se);
figure, subplot(1,2,1), imshowpair(I, bw), title('Segmented Image');
subplot(1,2,2), imshowpair(I, cjdata.tumorMask), title('Original Segmented Image');