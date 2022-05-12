function [pyr] = GetPyramids(image,levels)
%GETPYRAMIDS Summary of this function goes here
%   Detailed explanation goes here
pyr = cell(1,levels);
pyr{1} = image;

for i = 2:levels
    pyr{i} = impyramid(pyr{i-1},'reduce');
end

