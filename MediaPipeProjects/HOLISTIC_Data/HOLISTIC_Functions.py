import glob
import os
from path import Path
from threading import Thread
import pandas as pd
import mediapipe as mp
import cv2
from time import time, sleep
import csv
import sys
from queue import Queue
import numpy as np
# Functions for the files.


class ThreadVideoProcess:
    def __init__(self, video_id=""):
        self.video_id = video_id

        self.vid_cap = cv2.VideoCapture(video_id)
        if self.vid_cap.isOpened() is False:
            print("[Exiting]: Error loading the video.")

        self.grabbed, self.frame = self.vid_cap.read()
        if self.grabbed is False:
            print("No more frames")
            exit(0)

        self.stopped = True

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vid_cap.read()
            if self.grabbed is False:
                print("[Exiting], no more frames")
                self.stopped = True
                break
        self.vid_cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class PyImgVidStream:
    def __init__(self, path, queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.Q = Queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True


def total_frame(video):
    cap = cv2.VideoCapture(video)
    totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = totalframes / fps
    delay = fps/2
    print(duration)
    print(fps)
    print(delay)
    print(totalframes)


def file_wipe(directory):
    # file deletion function.
    for file in glob.glob(directory):
        os.remove(file)


def frame_maker(file, variable_to_follow):
    # file creation function.
    with open(Path(f"landmark_data/{file}"), "a", newline="") as f:
        writer = csv.writer(f)
        Thread(target=writer.writerow(variable_to_follow), args=())


def holistic_mo_cap_mp(video):
    # mediapipe motion capture function.
    # mp dependencies.
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    # body of motion capture.
    # video = the video to use, pass as function argument.
    cap = ThreadVideoProcess(video_id=video)
    sleep(1.0)
    # main body of tracking code.
    with mp_holistic.Holistic(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8) as holistic:
        num_frames_processed = 0
        cap.start()
        while True:
            image = cap.read()
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            num_frames_processed += 1
            print(num_frames_processed)
            # real time data extraction.
            # if len(list(results.left_hand_landmarks.landmark)) != 0:
            #     frame_maker("left_hand.csv", list(results.left_hand_landmarks.landmark))
            # if len(list(results.right_hand_landmarks.landmark)) != 0:
            #     frame_maker("right_hand.csv", list(results.right_hand_landmarks.landmark))
            # if len(list(results.face_landmarks.landmark)) != 0:
            #     frame_maker("face.csv", list(results.face_landmarks.landmark))
            frame_maker("pose.csv", list(results.pose_landmarks.landmark))
            sleep(0.03)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.stop()


file_wipe("landmark_data/*")
total_frame("Holistic_video/IMG_5852.MOV")
start = time()
holistic_mo_cap_mp("Holistic_video/IMG_5852.MOV")
end = time()
print(f"End of process! {str(end - start)}")
