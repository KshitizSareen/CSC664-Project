function [distances] = GetEuclideanDistance(ActualX,ActualY,MeasuredX,MeasuredY)
%GETEUCLIDEANDISTANCE Summary of this function goes here
distances = zeros(1,length(ActualX));

for i = 1:length(ActualX)
    distance = ((ActualY(i)-MeasuredY(i))^2)+((ActualX(i)-MeasuredX(i))^2);
    distances(i)=distance;
%   Detailed explanation goes here
end

