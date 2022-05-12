clear all;
close all;

videoReader = VideoReader('Video.mp4')
videoPlayer= vision.VideoPlayer;
frameRate = videoReader.FrameRate;
oldFrame = readFrame(videoReader);

for i =1:8
    M = zeros(4,12);
    for j =1: 12
        imshow(oldFrame);
        h_rect = drawrectangle();
        pos_rect = h_rect.Position;
        pos_rect = round(pos_rect);
        M(1,j) = pos_rect(1);
        M(2,j) = pos_rect(3);
        M(3,j) = pos_rect(2);
        M(4,j) = pos_rect(4);
        disp(pos_rect);
    end
    writematrix(M,strcat('M',num2str(i),'.xls'));
end



%{
while hasFrame(videoReader)
    newFrame = readFrame(videoReader);
    disp(newFrame);
    pause(1/frameRate);
    step(videoPlayer,newFrame);
end
%}