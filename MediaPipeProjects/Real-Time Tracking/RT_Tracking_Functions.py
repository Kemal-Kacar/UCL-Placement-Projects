import struct
import time
from glob import glob
from os import remove
import cv2
from path import Path
import csv
from threading import Thread
import mediapipe as mp
from time import sleep
import pickle
import pandas as pd
from timeit import default_timer as timer
from datetime import datetime
import numpy as np

                    ### WHERE I LEFT OFF ###
# TRYING TO APPENDED WRITE TO BINARY FILE FOR THE REAL-TIME TRACKING.
# SPECIFICALLY, AT 62-65 LINES FOR FRAME-MAKER FUNCTION.

#   working function. here for safekeeping
# def frame_maker(file, variable_to_follow, f_num):
#     # file creation function.
#     with open(Path(f"landmark_data/{file}/frame{f_num}.csv"), "w") as f:
#         dataframe = pd.DataFrame(list(variable_to_follow))
#         dataframe.to_csv(f, index=False)

# working write to binary function. same as above but here for reference
# for when I inevitably fuck up and mald for hours.
# def frame_maker(file, variable_to_follow, f_num):
#     # file creation function.
#     with open(Path(f"landmark_data/{file}/frame{f_num}.dat"), "wb") as f:
#         dataframe = list(variable_to_follow)
#         date = datetime.now().timestamp()
#         pickle.dump(dataframe, f)

m_list = []


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


def file_wipe(directory):
    # file deletion function.
    for file in glob(directory):
        remove(file)


def frame_maker(file, variable_to_follow, f_num):
    # file creation function.
    with open(Path(f"landmark_data/{file}/frame.dat"), "wb") as f:
        dataframe = list(variable_to_follow)
        date = datetime.now().timestamp()
        pickle.dump(dataframe, f)


def location_extractor(markers):
    m_list.append(time.time())
    # with open(Path(f"landmark_data/{file}/loc.dat"), "ab") as f:
    for loc in range(len(markers)):
        m_list.append(markers[loc].x)
        m_list.append(markers[loc].y)
        m_list.append(markers[loc].z)


def holistic_mo_cap_mp(video):
    start_time = timer()
    # mediapipe motion capture function.
    # mp dependencies.
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # body of motion capture.
    # video = the video to use, pass as function argument.
    cap = cv2.VideoCapture(video)
    sleep(1.0)
    # main body of tracking code.
    with mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8) as face_mesh:
        num_frames_processed = 0
        while cap.isOpened():
            success, image = cap.read()
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    num_frames_processed += 1
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
            # real time data extraction
                    if cap.isOpened():
                        print(num_frames_processed)
                        frame_maker("face", results.multi_face_landmarks[0].landmark, num_frames_processed)
                        location_extractor(results.multi_face_landmarks[0].landmark)
                    else:
                        continue

            # frame_maker("left_hand.csv", list(results.left_hand_landmarks.landmark))
            # frame_maker("right_hand.csv", list(results.right_hand_landmarks.landmark))

            # frame_maker("pose.csv", list(results.pose_landmarks.landmark))
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            # nmarkers = len( results.multi_face_landmarks[0].landmark)
            # xtracted = list()
            # for ii in range(0, nmarkers):
            #     xtracted.append(results.multi_face_landmarks[0].landmark[ii].x)
            #     xtracted.append(results.multi_face_landmarks[0].landmark[ii].y)
            #     xtracted.append(results.multi_face_landmarks[0].landmark[ii].z)
            #
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        end_time = timer()
        print(end_time-start_time)


file_wipe("landmark_data/face/*")
holistic_mo_cap_mp(0)
csv_m_list = m_list
a = np.array(m_list, 'float32')
with open(Path(f"landmark_data/face/loc.dat"), "wb") as y:
    a.tofile(y)

with open(Path(f"landmark_data/face/markercsv.csv"), "w", newline="") as csvW:
    mlistdf = pd.DataFrame(csv_m_list)
    mlistdf.to_csv(csvW, index=False)

