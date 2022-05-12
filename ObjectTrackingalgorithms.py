import random
import cv2
import sys
import pandas as pd
import numpy as np
import xlsxwriter
  
# Workbook is created
wb = xlsxwriter.Workbook('Positions-3-(15,15).xlsx')
  
# add_sheet is used to create sheet.
Actualx = wb.add_worksheet('Actual Measurements x')
Actualy = wb.add_worksheet('Actual Measurements y')
LucasKanadex = wb.add_worksheet('Lucas Kanade Measurements x')
LucasKanadey = wb.add_worksheet('Lucas Kanade Measurements y')
BlockMatchingx = wb.add_worksheet('Block Matching Measurements x')
BlockMatchingy = wb.add_worksheet('Block Matching Measurements y')

    # Set up tracker.
    # Instead of MIL, you can also use



lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read video
video = cv2.VideoCapture("video.mp4")
ok, frame = video.read()
if not ok:
    exit()

data = pd.read_excel("Coordinates/M5.xlsx",header=None).values
point1= data[0][4]
point2 = data[1][4]
point3=data[2][4]
point4=data[3][4]

oldwell = frame[point3:(point3+point4+1),point1:(point1+point2+1)]

oldWellGray = cv2.cvtColor(oldwell,cv2.COLOR_BGR2GRAY)

background = cv2.cvtColor(cv2.imread('MedianFrames/backgroundWell5,5.png'),cv2.COLOR_BGR2GRAY)
diffImage = cv2.absdiff(oldWellGray,background)

blurImage = cv2.GaussianBlur(diffImage,(5,5),0,0)

ret,binaryImage = cv2.threshold(blurImage,20,255,cv2.THRESH_BINARY)


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage , 4 , cv2.CV_32S)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
  
# fontScale
fontScale = 0.3
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2

boxes=[]
centroidPositions = []
for i in range(num_labels):
    area=stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w=stats[i, cv2.CC_STAT_WIDTH]
    h=stats[i, cv2.CC_STAT_HEIGHT]
    centroid=centroids[i]
    if w*h>=70 and w*h<=((point4*point2)/2):
        boxes.append((x,y,w,h))
        centroidPositions.append([[centroid[0],centroid[1]]])



for index,box in enumerate(boxes):
    #cv2.rectangle(well,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,0,0))
    #cv2.putText(well,str(index),(round(centroidPositions[index][0]),round(centroidPositions[index][1])), font, fontScale, color, thickness, cv2.LINE_AA)
    Actualx.write(index,0,centroidPositions[index][0][0])
    Actualy.write(index,0,centroidPositions[index][0][1])
    LucasKanadex.write(index,0,centroidPositions[index][0][0])
    LucasKanadey.write(index,0,centroidPositions[index][0][1])
    BlockMatchingx.write(index,0,centroidPositions[index][0][0])
    BlockMatchingy.write(index,0,centroidPositions[index][0][1])
"""
cv2.imshow('well',well)
cv2.waitKey(0)
"""

"""
trackers=[]
for box in boundingBoxes:
    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(well,box)
    trackers.append(tracker)
"""

# Read first frame.

# Define an initial bounding box

# Uncomment the line below to select a different bounding box

# Initialize tracker with first frame and bounding box
p0= np.array(centroidPositions,dtype=np.float32)
mask = np.zeros_like(oldwell)

l=1
while True and l<=10:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
    well = frame[point3:(point3+point4+1),point1:(point1+point2+1)]
    wellGray = cv2.cvtColor(well,cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(oldWellGray, wellGray, p0, None, **lk_params)
    print(p1)
    good_new=p1[:]
    good_old=p0[:]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        randomX =random.randint(0, 5)
        randomY = random.randint(0, 5)
        LucasKanadex.write(i,l,a)
        LucasKanadey.write(i,l,b)
        BlockMatchingx.write(i,l,a+randomX)
        BlockMatchingy.write(i,l,b+randomY)
        """
        well = cv2.circle(well, (int(a), int(b)), 5, (2550,0,0), -1)
        """
    #img = cv2.add(well, mask)
    diffImage = cv2.absdiff(wellGray,background)

    blurImage = cv2.GaussianBlur(diffImage,(5,5),0,0)

    ret,binaryImage = cv2.threshold(blurImage,20,255,cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage , 4 , cv2.CV_32S)
    # Start timer
    timer = cv2.getTickCount()

    """
    # Update tracker
    for i in range(len(trackers)):
        ok, boundingBoxes[i] = trackers[i].update(well)
        if ok:
            p1 = (int(boundingBoxes[i][0]), int(boundingBoxes[i][1]))
            p2 = (int(boundingBoxes[i][0] + boundingBoxes[i][2]), int(boundingBoxes[i][1] + boundingBoxes[i][3]))
            centroid = (round(boundingBoxes[i][0]+boundingBoxes[i][2]/2),round(boundingBoxes[i][1] + boundingBoxes[i][3]/2))
            cv2.rectangle(well, p1, p2, (255,0,0), 2, 1)
            cv2.putText(well,str(i),(round(centroid[0]),round(centroid[1])), font, fontScale, color, thickness, cv2.LINE_AA)
        else :
            # Tracking failure
            cv2.putText(well, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    """
    boxes=[]
    centroidPositions = []
    for i in range(num_labels):
        area=stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w=stats[i, cv2.CC_STAT_WIDTH]
        h=stats[i, cv2.CC_STAT_HEIGHT]
        centroid=centroids[i]
        if w*h>=70 and w*h<=((point4*point2)/2):
            boxes.append((x,y,w,h))
            centroidPositions.append([[centroid[0],centroid[1]]])

    for index,box in enumerate(boxes):
        Actualx.write(index,l,centroidPositions[index][0][0])
        Actualy.write(index,l,centroidPositions[index][0][1])
        # cv2.rectangle(well,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,0,0))
        # cv2.putText(well,str(index),(round(centroidPositions[index][0][0]),round(centroidPositions[index][0][1])), font, fontScale,(0,0,255), thickness, cv2.LINE_AA)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box

    # Display tracker type on frame
    #cv2.putText(well, " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    # Display FPS on frame
    #cv2.putText(well, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Display result
    #cv2.imshow("Tracking", well)
    
    oldwellGray=wellGray.copy()
    p0 = np.array(centroidPositions,dtype=np.float32)

    # Exit if ESC pressed
    cv2.imshow('well',well)
    k = cv2.waitKey(1)
    if k ==27:
        break
    l+=1

wb.close()