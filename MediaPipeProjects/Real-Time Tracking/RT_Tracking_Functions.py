# OS based imports.
from glob import glob
from os import remove
import cv2
from path import Path
from threading import Thread
# Time and tracking based imports.
from time import sleep
import time
from datetime import datetime
from timeit import default_timer as timer
# Dependencies and mediapipe imports.
import mediapipe as mp
import pickle
import numpy as np
import csv

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
# make a "README" file including any identifiable information.


class ThreadVideoProcess:
    """
    Integrating the video processing into a single thread.
    Currently not very useful as it is simply too fast to extract data.
    May have changed due to the optimisations done for real-time processing procedures.
    """
    def __init__(self, video_id=""):
        self.video_id = video_id

        self.vid_cap = cv2.VideoCapture(video_id)  # Integrate the cv2 capture method.
        if self.vid_cap.isOpened() is False:
            print("[Exiting]: Error loading the video.")

        self.grabbed, self.frame = self.vid_cap.read()  # Read the frame information.
        if self.grabbed is False:
            print("No more frames")
            exit(0)

        self.stopped = True  # Class integrated cv2.release function.

        self.t = Thread(target=self.update, args=())  # Threading line for the update function.
        self.t.daemon = True

    def start(self):  # cv2 isOpened functionality.
        self.stopped = False
        self.t.start()

    def update(self):  # Frame updater for the threading functionality.
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vid_cap.read()
            if self.grabbed is False:
                print("[Exiting], no more frames")
                self.stopped = True
                break
        self.vid_cap.release()

    def read(self):  # Frame reader functionality shadowing cv2.read function.
        return self.frame

    def stop(self):  # cv2.release function.
        self.stopped = True


def file_wipe(directory):
    # file deletion function.
    # Used to empty the frame files or any files.
    # Simply input the path of file, absolute and relative works.
    for file in glob(directory):
        remove(Path(file))


def frame_maker(file, variable_to_follow, f_num):
    # file creation function.
    # Currently underutilised. Function file to write binary data.
    # Can be made into frame-by-frame data creation/collection.
    # just add {f_num} after the frame string.
    with open(Path(f"landmark_data/{file}/frame.dat"), "wb") as f:
        dataframe = list(variable_to_follow)
        date = datetime.now().timestamp()
        pickle.dump(dataframe, f)


def location_extractor(markers):
    # extracts the landmark information onto an array.
    # takes the x/y/z information of each marker and appends relative to how many markers there are.
    for loc in range(len(markers)):
        m_list.append(markers[loc].x)
        m_list.append(markers[loc].y)
        m_list.append(markers[loc].z)


def holistic_mo_cap_mp(video):
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
    with mp_face_mesh.FaceMesh(  # Tweak these numbers for more/less accurate tracking.
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8) as face_mesh:
        # Visually show on the command window of the frame iteration. Can be deleted for speed.
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
            # a lot of these mp functions I am not too knowledgeable. Leave them be.
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
                        m_list.append(int(time.perf_counter_ns())/1000000)  # track the time and append to array.
                        # Call "frame_maker" function to make frame information files. pure dump function.
                        frame_maker("face", results.multi_face_landmarks[0].landmark, num_frames_processed)
                        # Call "location_extractor" function to extract specific landmark information
                        # as annotated on the function itself.
                        location_extractor(results.multi_face_landmarks[0].landmark)
                        # Show the time in window for us to see the change. Redundant, delete for speed.
                        print((time.perf_counter_ns())/1000000)
                    else:
                        continue
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


# Main calling section, all code happens through these lines.
# BE CAREFUL OF THIS LINE, I HAVE IT FOR TESTING PURPOSES, DO NOT RUN IF ACTUALLY USING.
file_wipe("landmark_data/face/*")                     # Call file_wipe and delete the directory.
m_list = []                                           # empty array to fill with the information.
start_time = timer()  # start the time. Just for visualisation, can be deleted to get a fraction of speed.
holistic_mo_cap_mp(0)                             # Start motion capture.
end_time = timer()
a = np.array(m_list, 'float32')                       # convert the binary datafile into np.array for better extraction.
# now = datetime.now()
file_date = str(datetime.now().strftime("_%d_%m_%Y-%H_%M_%S_"))
binary_file = f"landmark_data/face/{file_date}loc.dat"
with open(binary_file, "wb") as y:  # write the np.array list to the file.
    a.tofile(y)

with open(Path(f"landmark_data/face/{file_date}README_loc.txt"), "w") as readme:
    readme.write(f"Tracked information: x,y,z coordinates and time at the start of tracking. \n"
                 f"Used Face-Tracking solutions of mediapipe. \n"
                 f"File format: float32, little-endian, .dat file. \n"
                 f"Each frame is 1435 (1 for time, 1434 for x/y/z, 478 each) data long. \n"
                 f"Number of processed frames: {len(m_list)/1435} \n"
                 f"Number of total markers: {(1435-1)/3} \n"
                 f"Total time taken: {end_time-start_time} \n")
