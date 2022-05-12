import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
_,frame=cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

backgroundImage=cv2.imread('backgroundImage.png')

backgroundGray=cv2.cvtColor(backgroundImage, cv2.COLOR_BGR2GRAY)

subtractedImage = old_gray - backgroundGray

cv2.imshow('window',subtractedImage)
cv2.waitKey(0)