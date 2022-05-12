clear all;
close all;



%{
for i=4:6
    for j=4:6
        ReadData(i,j,1000);
    end
end

return;
%}


[ActualX44,ActualY44,ActualWidth44,ActualHeight44] = ReadMatrixes(4,4);
[ActualX45,ActualY45,ActualWidth45,ActualHeight45] = ReadMatrixes(4,5);
[ActualX46,ActualY46,ActualWidth46,ActualHeight46] = ReadMatrixes(4,6);
[ActualX54,ActualY54,ActualWidth54,ActualHeight54] = ReadMatrixes(5,4);
[ActualX55,ActualY55,ActualWidth55,ActualHeight55] = ReadMatrixes(5,5);
[ActualX56,ActualY56,ActualWidth56,ActualHeight56] = ReadMatrixes(5,6);
[ActualX64,ActualY64,ActualWidth64,ActualHeight64] = ReadMatrixes(6,4);
[ActualX65,ActualY65,ActualWidth65,ActualHeight65] = ReadMatrixes(6,5);
[ActualX66,ActualY66,ActualWidth66,ActualHeight66] = ReadMatrixes(6,6);


Coordinates =cell(1,9);

k=1;
for i=4:6
    for j=4:6
        [top,bottom,left,right]=ReadCoordinates(i,j);
        Coordinates{1,k}=[top,bottom,left,right];
        k=k+1;
    end
end

videoReader = VideoReader('Video.mp4');
frameRate = videoReader.FrameRate;
dt = 1/frameRate;
frame = readFrame(videoReader);
N=1000;

wells=cell(1,9);
wellData=cell(1,9);

wellData{1,1}=LoadFishData(frameRate,ActualX44,ActualY44,ActualWidth44,ActualHeight44,N);
wellData{1,2}=LoadFishData(frameRate,ActualX45,ActualY45,ActualWidth45,ActualHeight45,N);
wellData{1,3}=LoadFishData(frameRate,ActualX46,ActualY46,ActualWidth46,ActualHeight46,N);

wellData{1,4}=LoadFishData(frameRate,ActualX54,ActualY54,ActualWidth54,ActualHeight54,N);
wellData{1,5}=LoadFishData(frameRate,ActualX55,ActualY55,ActualWidth55,ActualHeight55,N);
wellData{1,6}=LoadFishData(frameRate,ActualX56,ActualY56,ActualWidth56,ActualHeight56,N);

wellData{1,7}=LoadFishData(frameRate,ActualX64,ActualY64,ActualWidth64,ActualHeight64,N);
wellData{1,8}=LoadFishData(frameRate,ActualX65,ActualY65,ActualWidth65,ActualHeight65,N);
wellData{1,9}=LoadFishData(frameRate,ActualX66,ActualY66,ActualWidth66,ActualHeight66,N);

disp(wellData);


fig=figure(1);

for i=1:1000
    frame=readFrame(videoReader);
    for j = 1:9
        Coordinate=Coordinates{1,j};
        wells{1,j}=frame(Coordinate(1)+(0:Coordinate(2)),Coordinate(3)+(0:Coordinate(4)));
        subplot(3,3,j);
        imshow(wells{1,j});
        hold on;
        fishData=wellData{1,j};
        for k = 1:8
            fish=fishData{1,k};
             measurementX=fish.ActualMeasurements(1,i);
             measurementY=fish.ActualMeasurements(2,i);
             width=fish.BoxSizes(1,i);
             height=fish.BoxSizes(2,i);
        
        
             predict(fish.filter,dt);
             fish.estimateStates(:,k) = correct(fish.filter,[measurementX;measurementY]);
             xPosition = fish.estimateStates(1,k); %measurementX; fish.estimateStates1(1,k);
             yPosition = fish.estimateStates(3,k); % measurementY;
             coordinates=[xPosition-(width/2) yPosition-(height/2) width height];
             rectangle('Position',coordinates,'EdgeColor',fish.color);
        end
    end
    pause(0.001);
end

close(fig);
