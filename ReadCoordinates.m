function [top,bottom,left,right] = ReadCoordinates(row,column)
%READCOORDINATES Summary of this function goes here
%   Detailed explanation goes here
M = readmatrix(strcat("Coordinates/M",num2str(row),".xlsx"));
top=M(3,column);
bottom=M(4,column);
left=M(1,column);
right=M(2,column);
end

