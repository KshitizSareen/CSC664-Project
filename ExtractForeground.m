clear all;
close all;


videoReader = VideoReader('Video.mp4');
l=0;
frameRate = videoReader.FrameRate;

startRow = 4;
endRow = 6;
startCol = 4;
endCol = 6;
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
            grayFrame = im2double(im2gray(well));
            grayBackground = im2double(im2gray(backgroundImage));
            diffImage = imabsdiff(grayBackground,grayFrame);
            blurImage = medfilt2(diffImage);
            T = graythresh(blurImage);
            binaryImage = imbinarize(blurImage,T);
            CC = bwconncomp(binaryImage,4);
            S = regionprops(CC,'Centroid');
            boxes= regionprops(CC,'BoundingBox');
            fish=[];
            for k = 1 : length(boxes)
                s= struct;
                boundingBox = boxes(k).BoundingBox;
                s.left = boundingBox(1);
                s.top = boundingBox(2);
                s.width = boundingBox(3);
                s.height = boundingBox(4);
                point = S(k).Centroid;
                s.point = point;
                fish = [fish,s];
            end
            if l >=1 
                opticalFlowPoints = ComputeOpticalFlow(wells{i,j},grayFrame,previousPoints{i,j},2,10,15,0.03);
                subplot(endRow-startRow+1,endCol-startCol+1,p);
                imshow(grayFrame);
                hold on;
                velocityX =[];
                velocityY = []
                startingPointsX = [];
                startingPointsY = [];
                for k = 1:length(opticalFlowPoints)
                    endingPoint = opticalFlowPoints(k).point;
                    startingPoint = previousPoints{i,j}(k).point;
                    startingPointsX = [startingPointsX,startingPoint(1)];
                    startingPointsY = [startingPointsY,startingPoint(2)];
                    velocityPoint = [endingPoint(1)-startingPoint(1),endingPoint(2)-startingPoint(2)];
                    velocityX = [velocityX,velocityPoint(1)]; 
                    velocityY = [velocityY,velocityPoint(2)]; 
                end
                quiver(startingPointsX,startingPointsY,velocityX,velocityY,1,'r');
            end
            previousPoints{i,j} = fish;
            wells{i,j} = grayFrame;
            p=p+1;

        end
    end
    l=l+1;
    pause(0.00000001);
end
close(fig);


predict