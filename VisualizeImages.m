clear;
folder='Data';
mainFolder=dir(folder);
imageIndex=2626;
imageName=mainFolder(imageIndex).name;
path=fullfile(folder,imageName);
load(path);
xi=cjdata.tumorBorder(1:2:end);
yi=cjdata.tumorBorder(2:2:end);
image=cjdata.image;
subplot(1,3,1), imshow(image, [])
subplot(1,3,2), imshow(image, []), hold on, plot(xi, yi, 'LineWidth', 1.5), hold off
subplot(1,3,3), imshowpair(image, cjdata.tumorMask)