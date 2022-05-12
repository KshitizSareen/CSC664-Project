function [points] = ComputeOpticalFlow(imageOne,imageTwo,points,levels,iterations,windowSize,threshold)
%COMPUTEOPTICALFLOW Summary of this function goes here
%   Detailed explanation goes here

imageOnePyramids = GetPyramids(imageOne,levels);
imageTwoPyramids = GetPyramids(imageTwo,levels);
h = fspecial('sobel');

halfWindow = floor(windowSize/2);

for i = 1: length(points)
    points(i).point = points(i).point / (2^levels);
end

for i = levels:-1:1
    imageOne=imageOnePyramids{i};
    imageTwo=imageTwoPyramids{i};

    imageOneX = imfilter(imageOne,h','replicate');
    imageOneY = imfilter(imageOne,h,'replicate');
    [M N] =size(imageOne);

    for j = 1: length(points)
        point = points(j).point;
        x = point(1)*2;
        y = point(2)*2;
        l = x - halfWindow;
        t = y - halfWindow;
        r = x + halfWindow;
        b = y + halfWindow;

        [xCoordinates, yCoordinates] = meshgrid(l:r,t:b);

        fl = floor(l);
        ft = floor(t);
        cr = ceil(r);
        cb = ceil(b);
        iX = fl:cr;
        iY = ft:cb;

        if fl>=1 && ft>=1 && cr<=N && cb<=M
            Ix = interp2(iX,iY,imageOneX(iY,iX),xCoordinates,yCoordinates);
            Iy = interp2(iX,iY,imageOneY(iY,iX),xCoordinates,yCoordinates);
            I1 = interp2(iX,iY,imageOne(iY,iX),xCoordinates,yCoordinates);

            for k =1:iterations
                l = x - halfWindow;
                t = y - halfWindow;
                r = x + halfWindow;
                b = y + halfWindow;
        
                [xCoordinates, yCoordinates] = meshgrid(l:r,t:b);
        
                fl = floor(l);
                ft = floor(t);
                cr = ceil(r);
                cb = ceil(b);
                iX = fl:cr;
                iY = ft:cb;
                if fl>=1 && ft>=1 && cr<=N && cb<=M
                    It = interp2(iX,iY,imageTwo(iY,iX),xCoordinates,yCoordinates) - I1;
                    A = [Ix(:),Iy(:)];
                    vel = A\It(:);
                    x=x+vel(1);
                    y=y+vel(2);
                    if max(abs(vel))<threshold
                        break
                    end
                end
            end
        end
        points(j).point = [x y];

    end
end

end

