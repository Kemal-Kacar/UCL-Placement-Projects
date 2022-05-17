import cv2
from CountsPerSec import CountsPerSec
from MediaPipeProjects.FULLBODY_Data.VideoGet import VideoGet
import time
import mediapipe as mp
import pandas as pd
from path import Path
import os
import glob
from WriteJson import ToJSON
import csv
from threading import Thread
#import modin.pandas as pd
#import ray
#ray.init()
# from VideoShow import VideoShow


# def putIterationsPerSec(frame, iterations_per_sec):
#     """
#     Add iterations per second text to lower-left corner of a frame.
#     """
#
#     cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
#         (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
#     return frame
#
#
# def noThreading (source=0):
#
#     cap = cv2.VideoCapture(source)
#     cps = CountsPerSec().start()
#
#     while True:
#         (grabbed, frame) = cap.read()
#         if not grabbed or cv2.waitKey(1) == ord("q"):
#             break
#
#         frame = putIterationsPerSec(frame, cps.countsPerSec())
#         cv2.imshow("Video", frame)
#         cps.increment()
#

cap = cv2.VideoCapture("IMG_5852.MOV")
totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalframes)
Demo_FrameFiles = glob.glob("Demo_Frames/*")
for jsF in Demo_FrameFiles:
    os.remove(jsF)
Demo_frames = "Demo_Frames/"


def threadVideoGet(source=""):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
    n_frame = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    video_getter = VideoGet(source).start()
    # cps = CountsPerSec().start()
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while True:
            if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
                video_getter.stop()
                break
            image = video_getter.frame
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if n_frame < int(totalframes):
                frame_maker(Demo_frames, list(results.pose_world_landmarks.landmark))
                n_frame += 1
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        video_getter.stop()

# def extractor(x, y, z):
#     with open("random.csv", "a") as extractorData:
#         for LMN in range(results.landmark):
#             df = pd.DataFrame(extractorData)
#             df[x] = [results.pose_world_landmarks.landmark["x"]]
#             df[y] = [results.pose_world_landmarks.landmark["y"]]
#             df[z] = [results.pose_world_landmarks.landmark["z"]]
#             return df


def frame_maker(frame_filepath, variable_to_follow):
    n_frame = 0
    # creates json files from the taken frames.
    f_filepath = Path(f"{frame_filepath}frame{n_frame}.json")
    variable_to_follow = pd.DataFrame(variable_to_follow)
    Thread(target=variable_to_follow.to_json(f_filepath, indent=0)).start()


# header = ["landmark", "x", "y", "z", "distance"]
# with open("random.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(head for head in header)

start = time.time()
print("start of video!")
threadVideoGet("IMG_5852.MOV")
end = time.time()
print(end - start)
