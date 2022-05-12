import numpy as np
import pandas as pd
import cv2


for i in range(4,7):
    data = pd.read_excel("Coordinates/M"+str(i)+".xlsx",header=None).values
    for j in range(4,7):
        point1= data[0][j-1]
        point2 = data[1][j-1]
        point3=data[2][j-1]
        point4=data[3][j-1]
        video = cv2.VideoCapture('Video.mp4')
        k=0
        Array=[]
        while 1 and k<15000:
            _,frame=video.read()
            if not _:
                break
            well=frame[point3:(point3+point4+1),point1:(point1+point2+1)]
            Array.append(well)
            k+=1
        medianArray=np.median(Array,axis=0).astype(dtype=np.uint8)
        cv2.imwrite('MedianFrames/backgroundWell'+str(i)+','+str(j)+'.png',medianArray)
