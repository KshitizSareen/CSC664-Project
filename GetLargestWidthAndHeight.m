clear all;
close all;

minWidth = 200;
minHeight = 200;

for i=1:8
    for j=1:12
        image=imread(strcat("MedianFrames/backgroundWell",num2str(i),',',num2str(j),".png"));
        disp(size(image));
        [width, height] = size(image);
        minWidth=min(minWidth,width);
        minHeight=min(minHeight,height);
    end
end

disp(minWidth);
disp(minHeight);