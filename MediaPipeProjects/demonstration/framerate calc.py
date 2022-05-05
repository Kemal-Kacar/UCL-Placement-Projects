import cv2
import time

video = "sub-002_ses-002_task-hokey_cokey_run-005_video.mp4"

cap= cv2.VideoCapture(video)
totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
framespersecond= int(cap.get(cv2.CAP_PROP_FPS))

print("The total number of frames in this video is ", framespersecond)
print(totalframes)
