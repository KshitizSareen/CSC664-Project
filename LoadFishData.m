function [fishData] = LoadFishData(frameRate,ActualX,ActualY,ActualWidth,ActualHeight,N)
%LOADFISHDATA Summary of this function goes here
%   Detailed explanation goes here

dt = 1/frameRate;
initialCovariance = [(dt^3/3) (dt^2)/2 0 0;(dt^2)/2 (dt) 0 0;0 0 (dt^3/3) (dt^2)/2; 0 0 (dt^2)/2 (dt)];
fishData = cell(1,8);

for i=1:8
    fish=struct;
    InitialPositionX=ActualX(i,1);
    InitialPositionY=ActualY(i,1);
    fish.initialPosition=[InitialPositionX;0;InitialPositionY;0];
    fish.filter=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 0.01; 0; 0.01]),MeasurementFcn=@measureModel,MeasurementNoise=diag([0.01;0.01]));
    fish.estimateStates=NaN(4,N);
    fish.estimateStates(:,1) = fish.filter.State;
    fish.color=randomColor();
    fish.ActualMeasurements=[ActualX(i,:);ActualY(i,:)];
    fish.BoxSizes=[ActualWidth(i,:);ActualHeight(i,:)];
    fishData{1,i}=fish;
end
end

function stateNext = stateModel(state,dt)
    F = [1 dt 0  0; 
         0  1 0  0;
         0  0 1 dt;
         0  0 0  1];
    stateNext = F*state;
end

function z = measureModel(state)
    z = [state(1)+state(2);state(3)+state(4)];
end

function color = randomColor()
    color = rand(1,3);
end