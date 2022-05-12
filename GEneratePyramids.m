videoReader = VideoReader('Video.mp4');
frameRate = videoReader.FrameRate;
frame = readFrame(videoReader);


levels = 4;
pyr = cell(1,levels);

M = readmatrix(strcat("Coordinates/M",num2str(5),".xlsx"));
array =[M(3,5), M(4,5), M(1,5), M(2,5)];
well = frame(array(1)+(0:array(2)),array(3)+(0:array(4)));
figure;
imshow(well);
disp(size(well));
pyr{1} = well;
for i = 2:levels
    figure;
    pyr{i} = impyramid(pyr{i-1},'reduce');
    imshow(pyr{i});
    disp(size(pyr{i}));
end