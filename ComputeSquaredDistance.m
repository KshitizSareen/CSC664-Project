function [distance] = ComputeSquaredDistance(pointOne,pointTwo)
%COMPUTESQUAREDDISTANCE Summary of this function goes here
%   Detailed explanation goes here
xOne = pointOne(1);
yOne = pointOne(2);

xTwo = pointTwo(1);
yTwo = pointTwo(2);

distance = (xTwo-xOne)^2 + (yTwo-yOne)^2;
end

