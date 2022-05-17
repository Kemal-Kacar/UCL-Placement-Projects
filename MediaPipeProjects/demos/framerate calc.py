import cv2
import time

video = 0

cap = cv2.VideoCapture(video)
totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
framespersecond= int(cap.get(cv2.CAP_PROP_FPS))

print("The total number of frames in this video is ", framespersecond)
print(totalframes)
cap = cv2.VideoCapture(0)
totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(int(totalframes))