import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/kshitiz/CSC 664 Materials/Optical Flow Project/ZebraFIshOneWell.mov')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )
_,frame=cap.read()
frames=[]
for i in range(length):
    _,frame=cap.read()
    if not _:
        break
    frames.append(frame)

medianFrame = np.median(frames,axis=0).astype(dtype=np.uint8)
meanFrame = np.mean(frames,axis=0).astype(dtype=np.uint8)

cv2.imshow('medianBackground',medianFrame)
cv2.imwrite('zebraFishBackgroundImage.png',medianFrame)
cv2.imshow('meanBackground',meanFrame)
cv2.waitKey(0)

