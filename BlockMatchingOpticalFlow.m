function [Ans] = BlockMatchingOpticalFlow(img1,img2,points,windowSize)
%BLOCKMATCHINGOPTICALFLOW Summary of this function goes here
%   Detailed explanation goes here

halfWindow = windowSize/2;
[height, width] =size(img2);

Ans=[];
for i =1:length(points)
    blockHeight = points(i).height;
    blockWidth = points(i).width;
    x=points(i).point(1);
    y=points(i).point(2);
    if points(i).top>=1 && points(i).top+blockHeight<=height && points(i).left>=1 && points(i).left+blockWidth<=width
    block = img1(floor(points(i).top):floor(points(i).top)+floor(blockHeight),floor(points(i).left):floor(points(i).left)+floor(blockWidth));
    minBlock=struct;
    minBlock.SSD=inf;
    minBlock.point=[x y];
    for j = floor(y-halfWindow):ceil(y+halfWindow)
        for k=floor(x-halfWindow):ceil(x+halfWindow)
            if j>=1 && j+blockHeight<=height && k>=1 && k+blockWidth<=width
            blockTwo=img2(j:j+floor(blockHeight),k:k+floor(blockWidth));
            SSD=sum(sum(block-blockTwo).^2);
                if SSD<minBlock.SSD
                    minBlock.SSD=SSD;
                    minBlock.point = [j+blockHeight/2 k+blockWidth/2];
                end
            end
        end
    end
    Ans = [Ans,minBlock];
    end

end


end

