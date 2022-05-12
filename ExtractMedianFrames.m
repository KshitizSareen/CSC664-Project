clear all;
close all;

backgroundImage = imread('backgroundImage.png');

for i = 1:8
    M = readmatrix(strcat("Coordinates/M",num2str(i),".xlsx"));
    for j = 1:12
        array =[M(3,j), M(4,j), M(1,j), M(2,j)];
        well = backgroundImage(array(1)+(0:array(2)),array(3)+(0:array(4)));
        imwrite(well,strcat('MedianFrames/backgroundWell',num2str(i),',',num2str(j),'.png'));
    end
end