clear all;
close all;

ActualX=readmatrix('Positions-2-(15,15).xlsx','Sheet','Actual Measurements x','Range','A1:J1');
ActualX(isnan(ActualX))=0;
disp(length(ActualX));
ActualY=readmatrix('Positions-2-(15,15).xlsx','Sheet','Actual Measurements y','Range','A1:J1');
ActualY(isnan(ActualY))=0;
LucasKanadeX=readmatrix('Positions-2-(15,15).xlsx','Sheet','Lucas Kanade Measurements x','Range','A1:J1');
LucasKanadeX(isnan(LucasKanadeX))=0;
LucasKanadeY=readmatrix('Positions-2-(15,15).xlsx','Sheet','Lucas Kanade Measurements y','Range','A1:J1');
LucasKanadeY(isnan(LucasKanadeY))=0;
BlockMatchingX=readmatrix('Positions-2-(15,15).xlsx','Sheet','Block Matching Measurements x','Range','A1:J1');
BlockMatchingX(isnan(BlockMatchingX))=0;
BlockMatchingY=readmatrix('Positions-2-(15,15).xlsx','Sheet','Block Matching Measurements y','Range','A1:J1');
BlockMatchingY(isnan(BlockMatchingY))=0;

%{
f1= figure;
distances = GetEuclideanDistance(ActualX,ActualY,LucasKanadeX,LucasKanadeY);

plot(linspace(1,length(ActualX),length(ActualX)),distances);
ylim([0 0.01]);
xlabel('frames');
ylabel('Displacement');
title('Error in Tracking for Lucas Kanade Algorithm For 3 Pyramid and (15,15) window');
%}

f2= figure;
distances = GetEuclideanDistance(ActualX,ActualY,BlockMatchingX,BlockMatchingY);

plot(linspace(1,length(ActualX),length(ActualX)),distances);
xlabel('frames');
ylabel('Displacement');
title('Error in Tracking for Block Matching Algorithm for (15x15) window');
