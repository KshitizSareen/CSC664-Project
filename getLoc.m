function [boxLoc,centroidLoc] = getLoc(hrect)
% Copyright 2015-2016 The MathWorks, Inc.
bbox = hrect.Position;
boxLoc = round(bbox);
centroidLoc(1) = bbox(1) + (bbox(3)/2);
centroidLoc(2) = bbox(2) + (bbox(4)/2);
end
