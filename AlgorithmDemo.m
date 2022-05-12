function [] = AlgorithmDemo(row,column,N,plotRow,plotColumn,plotNumber)
%ALGORITHMDEMO Summary of this function goes here
%   Detailed explanation goes here
ActualX=load(strcat('ActualX',num2str(row),num2str(column),'.mat')).ActualX;
ActualY=load(strcat('ActualY',num2str(row),num2str(column),'.mat')).ActualY;
ActualWidth=load(strcat('ActualWidth',num2str(row),num2str(column),'.mat')).ActualWidth;
ActualHeight=load(strcat('ActualHeight',num2str(row),num2str(column),'.mat')).ActualHeight;





end

