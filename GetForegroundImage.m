function [outputArg1,outputArg2] = GetForeground(inputArg1,inputArg2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
detector = vision.ForegroundDetector('NumTrainingFrames',50,'InitialVariance',30*30,'NumGaussians',3);
end

