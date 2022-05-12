clear all;
close all;




videoReader = VideoReader('Video.mp4');
frameRate = videoReader.FrameRate;
dt = 1/frameRate;
frame = readFrame(videoReader);
fig=figure;
fishData = cell(1,8);
initialCovariance = [(dt^3/3) (dt^2)/2 0 0;(dt^2)/2 (dt) 0 0;0 0 (dt^3/3) (dt^2)/2; 0 0 (dt^2)/2 (dt)];
N=1000;


column = num2xlcol(N);

ActualX=readmatrix('NewPositions.xlsx','Sheet','Actual Measurements x','Range',strcat('A','1',':',column,'8'));
ActualY=readmatrix('NewPositions.xlsx','Sheet','Actual Measurements y','Range',strcat('A','1',':',column,'8'));
ActualWidth=readmatrix('NewPositions.xlsx','Sheet','Actual Measurements Width','Range',strcat('A','1',':',column,'8'));
ActualHeight=readmatrix('NewPositions.xlsx','Sheet','Actual Measurements Height','Range',strcat('A','1',':',column,'8'));

for i=1:8
    fish=struct;
    InitialPositionX=ActualX(i,1);
    InitialPositionY=ActualY(i,1);
    fish.initialPosition=[InitialPositionX;0;InitialPositionY;0];
    fish.filter1=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 0.01; 0; 0.01]),MeasurementFcn=@measureModel,MeasurementNoise=diag([0.01;0.01]));
    fish.filter2=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 15.0; 0; 15.0]),MeasurementFcn=@measureModel,MeasurementNoise=diag([0.01;0.01]));
    fish.filter3=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 30.0; 0; 30.0]),MeasurementFcn=@measureModel,MeasurementNoise=diag([0.01;0.01]));
    fish.filter4=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 0.01; 0; 0.01]),MeasurementFcn=@measureModel,MeasurementNoise=diag([15.0;15.0]));
    fish.filter5=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 0.01; 0; 0.01]),MeasurementFcn=@measureModel,MeasurementNoise=diag([30.0;30.0]));
    fish.filter6=trackingEKF(State=fish.initialPosition,StateCovariance=initialCovariance, StateTransitionFcn=@stateModel,ProcessNoise=diag([0; 30.0; 0; 30.0]),MeasurementFcn=@measureModel,MeasurementNoise=diag([30.0;30.0]));
    fish.estimateStates1=NaN(4,N);
    fish.estimateStates1(:,1) = fish.filter1.State;
    fish.estimateStates2=NaN(4,N);
    fish.estimateStates2(:,1) = fish.filter2.State;
    fish.estimateStates3=NaN(4,N);
    fish.estimateStates3(:,1) = fish.filter3.State;
    fish.estimateStates4=NaN(4,N);
    fish.estimateStates4(:,1) = fish.filter4.State;
    fish.estimateStates5=NaN(4,N);
    fish.estimateStates5(:,1) = fish.filter5.State;
    fish.estimateStates6=NaN(4,N);
    fish.estimateStates6(:,1) = fish.filter6.State;
    fish.color=randomColor();
    fishData{1,i}=fish;
end

M = readmatrix(strcat("Coordinates/M",num2str(5),".xlsx"));
array =[M(3,5), M(4,5), M(1,5), M(2,5)];

estimates1=zeros(2,N);
estimates1(:,1)=[InitialPositionX;InitialPositionY];

estimates2=zeros(2,N);
estimates2(:,1)=[InitialPositionX;InitialPositionY];

estimates3=zeros(2,N);
estimates3(:,1)=[InitialPositionX;InitialPositionY];

estimates4=zeros(2,N);
estimates4(:,1)=[InitialPositionX;InitialPositionY];

estimates5=zeros(2,N);
estimates5(:,1)=[InitialPositionX;InitialPositionY];

estimates6=zeros(2,N);
estimates6(:,1)=[InitialPositionX;InitialPositionY];

for k = 2 : N
    frame = readFrame(videoReader);
    well = frame(array(1)+(0:array(2)),array(3)+(0:array(4)));
    grayFrame = im2gray(well);
    imshow(grayFrame);
    hold on;
    %{
    subplot(2,3,1);
    title("Process Noise =0.01 and Measurement Noise = 0.01");
    imshow(grayFrame);
    hold on;

    subplot(2,3,2);
    title("Process Noise =1.50 and Measurement Noise = 0.01");
    imshow(grayFrame);
    hold on;

    subplot(2,3,3);
    title("Process Noise =3.00 and Measurement Noise = 0.01");
    imshow(grayFrame);
    hold on;

    subplot(2,3,4);
    title("Process Noise =0.01 and Measurement Noise = 1.50");
    imshow(grayFrame);
    hold on;

    subplot(2,3,5);
    title("Process Noise =0.01 and Measurement Noise = 3.00");
    imshow(grayFrame);
    hold on;

    subplot(2,3,6);
    title("Process Noise =3.00 and Measurement Noise = 3.00");
    imshow(grayFrame);
    hold on;
    %}
    for i=1:8
     fish=fishData{1,i};
     measurementX=ActualX(i,k);
     measurementY=ActualY(i,k);
     width=ActualWidth(i,k);
     height=ActualHeight(i,k);


     predict(fish.filter1,dt);
     fish.estimateStates1(:,k) = correct(fish.filter1,[measurementX;measurementY]);
     estimates1(:,k)=[fish.estimateStates1(1,k);fish.estimateStates1(3,k)];
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     rectangle('Position',coordinates,'EdgeColor',fish.color);

     
     %{
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     disp(coordinates);
     subplot(2,3,1);
     rectangle('Position',coordinates,'EdgeColor',fish.color);
     %}

     predict(fish.filter2,dt);
     fish.estimateStates2(:,k) = correct(fish.filter2,[measurementX;measurementY]);
     estimates2(:,k)=[fish.estimateStates2(1,k);fish.estimateStates2(3,k)];
     %{
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     disp(coordinates);
     subplot(2,3,1);
     rectangle('Position',coordinates,'EdgeColor',fish.color);
     %}

     predict(fish.filter3,dt);
     fish.estimateStates3(:,k) = correct(fish.filter3,[measurementX;measurementY]);
     estimates3(:,k)=[fish.estimateStates3(1,k);fish.estimateStates3(3,k)];
     %{
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     disp(coordinates);
     subplot(2,3,1);
     rectangle('Position',coordinates,'EdgeColor',fish.color);
     %}

     predict(fish.filter4,dt);
     fish.estimateStates4(:,k) = correct(fish.filter4,[measurementX;measurementY]);
     estimates4(:,k)=[fish.estimateStates4(1,k);fish.estimateStates4(3,k)];
     %{
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     disp(coordinates);
     subplot(2,3,1);
     rectangle('Position',coordinates,'EdgeColor',fish.color);
     %}

     predict(fish.filter5,dt);
     fish.estimateStates5(:,k) = correct(fish.filter5,[measurementX;measurementY]);
     estimates5(:,k)=[fish.estimateStates5(1,k);fish.estimateStates5(3,k)];
     %{
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     disp(coordinates);
     subplot(2,3,1);
     rectangle('Position',coordinates,'EdgeColor',fish.color);
     %}

     predict(fish.filter6,dt);
     fish.estimateStates6(:,k) = correct(fish.filter6,[measurementX;measurementY]);
     estimates6(:,k)=[fish.estimateStates6(1,k);fish.estimateStates6(3,k)];
     %{
     xPosition = fish.estimateStates1(1,k);
     yPosition = fish.estimateStates1(3,k);
     coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
     disp(coordinates);
     subplot(2,3,1);
     rectangle('Position',coordinates,'EdgeColor',fish.color);
     %}
    end
    pause(1/25);
end

close(fig);

fish=fishData{1,1};
xAxis=linspace(1,N,N);

figure(1);
yAxis1=estimates1(1,:);
plot(xAxis,yAxis1,"r",DisplayName="Estimated x position");
hold on;
yAxis3=ActualX(1,:);
plot(xAxis,yAxis3,"b",DisplayName="Actual x position");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion");
legend(Location="northoutside");
axis square


figure(2);
yAxis2=estimates1(2,:);
plot(xAxis,yAxis2,"g",DisplayName="Estimated y position");
hold on;
yAxis4=ActualY(1,:);
plot(xAxis,yAxis4,"y",DisplayName="Actual y position");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion");
legend(Location="northoutside");
axis square


%{
figure(2);
xAxis=linspace(1,N,N);
yAxis=(estimates2(1,:)-ActualX(1,:)).^2 + (estimates2(2,:)-ActualY(1,:)).^2;
%disp(fish.estimateStates1(1,:));
%disp(ActualX(1,:));
plot(xAxis,yAxis,"r");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion for process noise of 15 and measurement noise of 0.01");
axis square

figure(3);
xAxis=linspace(1,N,N);
yAxis=(estimates3(1,:)-ActualX(1,:)).^2 + (estimates3(2,:)-ActualY(1,:)).^2;
%disp(fish.estimateStates1(1,:));
%disp(ActualX(1,:));
plot(xAxis,yAxis,"r");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion for process noise of 30.0 and measurement noise of 0.01");
axis square

figure(4);
xAxis=linspace(1,N,N);
yAxis=(estimates4(1,:)-ActualX(1,:)).^2 + (estimates4(2,:)-ActualY(1,:)).^2;
%disp(fish.estimateStates1(1,:));
%disp(ActualX(1,:));
plot(xAxis,yAxis,"r");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion for process noise of 0.01 and measurement noise of 15.0");
axis square

figure(5);
xAxis=linspace(1,N,N);
yAxis=(estimates5(1,:)-ActualX(1,:)).^2 + (estimates5(2,:)-ActualY(1,:)).^2;
%disp(fish.estimateStates1(1,:));
%disp(ActualX(1,:));
plot(xAxis,yAxis,"r");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion for process noise of 0.01 and measurement noise of 30.0");
axis square


figure(6);
xAxis=linspace(1,N,N);
yAxis=(estimates6(1,:)-ActualX(1,:)).^2 + (estimates6(2,:)-ActualY(1,:)).^2;
%disp(fish.estimateStates1(1,:));
%disp(ActualX(1,:));
plot(xAxis,yAxis,"r");
xlabel("frames")
ylabel("difference in estimated and measured postions");
title("Estimated Motion vs Measured Motion for process noise of 30.0 and measurement noise of 30.0");
axis square
%}


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

function xlcol_addr=num2xlcol(col_num)
% col_num - positive integer greater than zero
    n=1;
    while col_num>26*(26^n-1)/25
        n=n+1;
    end
    base_26=zeros(1,n);
    tmp_var=-1+col_num-26*(26^(n-1)-1)/25;
    for k=1:n
        divisor=26^(n-k);
        remainder=mod(tmp_var,divisor);
        base_26(k)=65+(tmp_var-remainder)/divisor;
        tmp_var=remainder;
    end
    xlcol_addr=char(base_26); % Character vector of xlcol address
end