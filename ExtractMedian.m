clear all;
close all;


videoReader = VideoReader('Video.mp4')
videoPlayer= vision.VideoPlayer;
frameRate = videoReader.FrameRate;

duration = videoReader.Duration;


medianMatrix = zeros(796,1180,8000,'uint8');
k=1;
while hasFrame(videoReader) && k<=8000
    newFrame = readFrame(videoReader);
    grayFrame = im2gray(newFrame);
    medianMatrix(:,:,k)=grayFrame;
    k=k+1;
end

disp("Median Computation");
medianImage = median(medianMatrix,3);
imshow(medianImage);
imwrite(medianImage,'backgroundImage.png');

%{
while hasFrame(videoReader) && k<=3000
    newFrame = readFrame(videoReader);
    for i = 1:3
        M = readmatrix(strcat("Coordinates/M",num2str(i),".xlsx"));
        for j = 1:9
            array =[M(3,j), M(4,j), M(1,j), M(2,j)];
            well = newFrame(array(1)+(0:array(2)),array(3)+(0:array(4)));
            well =imresize(well,[100 100]);
            wellMatrix(:,:,i,j,k)=well;
        end
    end
    k=k+1;
end

medianImage = median(wellMatrix,5);
grayImage = mat2gray(medianImage(:,:,3,5));
imshow(im2uint8(grayImage));
%}
