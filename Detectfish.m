clear all;
close all;

videoReader = VideoReader('Video.mp4');
l=0;
frameRate = videoReader.FrameRate;

startRow = 4;
endRow = 4;
startCol = 4;
endCol = 4;
wells = cell(8,12);
previousPoints = cell(8,12);
fig=figure;

while hasFrame(videoReader)
    frame = readFrame(videoReader);
    p=1;
    for i = startRow:endRow
        M = readmatrix(strcat("Coordinates/M",num2str(i),".xlsx"));
        for j=startCol:endCol
            array =[M(3,j), M(4,j), M(1,j), M(2,j)];
            well = frame(array(1)+(0:array(2)),array(3)+(0:array(4)));
            backgroundImage = imread(strcat("MedianFrames/","backgroundWell",num2str(i),',',num2str(j),'.png'));
            grayFrame = im2gray(well);
            grayBackground = im2gray(backgroundImage);
            diffImage = imabsdiff(grayFrame,grayBackground);
            blurImage = imgaussfilt(diffImage);
            contrastDifference = imadjust(blurImage,[0 0.095], [0 1]);
            binaryImage = imbinarize(contrastDifference);
            CC = bwconncomp(binaryImage,4);
            S = regionprops(CC,'Centroid');
            boxes= regionprops(CC,'BoundingBox');
            areas = regionprops(CC,'Area');
            fish=[];
            subplot(1,2,1);
            hold on;
            imshow(binaryImage);
            for k = 1 : length(boxes)
                s= struct;
                boundingBox = boxes(k).BoundingBox;
                rectangle('Position',boundingBox,'EdgeColor','r');
                text(S(k).Centroid(1),S(k).Centroid(2),num2str(areas(k).Area),'Color','b');
            end
            subplot(1,2,2);
            imshow(grayFrame);
            pause(1/25);
        end
    end
end