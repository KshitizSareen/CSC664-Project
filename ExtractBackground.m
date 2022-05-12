clear all;
close all;


videoSource = VideoReader("Video.mp4");

detector = vision.ForegroundDetector('NumTrainingFrames',1000,'InitialVariance',30*30);
blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 125);

shapeInserter = vision.ShapeInserter('BorderColor','White');

videoPlayer = vision.VideoPlayer();
array = readmatrix("Coordinates/M4.xlsx");


while hasFrame(videoSource)
     frame  = readFrame(videoSource);
     well = frame(array(1)+(0:array(2)),array(3)+(0:array(4)));
     fgMask = detector(well);
     imshow(fgMask);
     bbox   = blob(fgMask);
     out    = shapeInserter(well,bbox);
     videoPlayer(out);
     pause(0.1);
end
    