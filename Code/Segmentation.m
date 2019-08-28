clear all;
close all;
clc;

seed = 50;
rng(seed);

load(strcat('../Data/2040.mat'));
img = cjdata.image;
img = uint8(255*mat2gray(img));
%img = imresize(img,[200,200]);

nrows = size(img,1);
ncols = size(img,2);
nColors = 4;

I = reshape(img, nrows*ncols, 1);

[id,c] = kmeans(I, nColors, 'distance', 'sqeuclidean', 'Replicates', 3);

% Label every pixel in tha image using results from K means
pixel_labels = reshape(id,nrows,ncols);

figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,nColors);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,nColors]);

for k = 1:nColors
    colors = img;
    colors(rgb_label(:,:,k) ~= k) = 0;
    segmented_images{k} = colors;
    subplot(1,nColors,k), imshow(segmented_images{k});
end


level = graythresh(segmented_images{4});
seg_img = im2bw(segmented_images{4}, level);

se = strel('disk',2);
seg_img = imclose(seg_img, se);
seg_img = imopen(seg_img, se);

figure, subplot(1,2,1), imshowpair(img, seg_img); title('Segmented Tumor');
subplot(1,2,2), imshowpair(img, cjdata.tumorMask); title('Original Image');
