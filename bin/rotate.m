%% Rotate png to upright position

clear, clc, close all

fromDir = '../data/Training_Image_unrot/';
toDir = '../data/Training_Image_png/'; 
data = dir(fromDir);
filePaths = {data.name};
for i = 1:length(filePaths)
	[pathstr,name,ext] = fileparts(filePaths{i});
	if (strcmp(ext,'.png'))
        input_image = imread(sprintf('%s%s', fromDir, filePaths{i}));
		if size(input_image, 3) == 3
            img = im2double(rgb2gray(input_image));
        else
            img = im2double(input_image);
        end
		[height, width] = size(img);

		% Binarize image
		imgBin = img < 0.9;

		% Calculate Hough transform
		thetaVec = -90 : 0.5 : 89.5;
		[H,theta,rho] = hough(imgBin, 'Theta', thetaVec);
		numPeaks = 40;
		thresh = 0.5*max(H(:));
		peaks = houghpeaks(H, numPeaks, 'Threshold', thresh);
		[thetaHist, X]= hist(theta(peaks(:,2)), thetaVec);
		[maxHist, maxIdx] = max(thetaHist);
		maxTheta = thetaVec(maxIdx)
		rotTheta = 90 + maxTheta;
		% Rotate image to upright orientation
		rot = @(I) imrotate(I, rotTheta, 'bilinear', 'loose');
		imgRot = rot(img);
		mask = rot(ones(size(img)));
		imgRot = imgRot + ~mask;
		path = sprintf('%s%s', toDir, filePaths{i});
		imwrite(imgRot, path)
	end
end
