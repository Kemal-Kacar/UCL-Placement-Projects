import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
import csv
import json
import os
import glob

# deletes all the data of the previous iteration to not have interference.
HAND_FrameFiles = glob.glob("Zero_HAND_Frames/*")
for ZjsF in HAND_FrameFiles:
    os.remove(ZjsF)
HAND_FrameFiles = glob.glob("One_HAND_Frames/*")
for ZjsF in HAND_FrameFiles:
    os.remove(ZjsF)
HAND_Landmarks = glob.glob("Zero_HAND_Landmark_CSVs/*")
for csvF in HAND_Landmarks:
    os.remove(csvF)
HAND_Landmarks = glob.glob("One_HAND_Landmark_CSVs/*")
for csvF in HAND_Landmarks:
    os.remove(csvF)

# mediapipe stuff, dependencies.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_motion = mp.solutions.hands.HandLandmark
frameCounter = 0
# the video file. Just change the "" area.
videoFile = "HAND_Video_data/sub-002_ses-002_task-hokey_cokey_run-005_video.mp4"

# change the () area to 0 if you want to use actual camera (in order of computer camera).
cap = cv2.VideoCapture(videoFile)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            if frameCounter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                Zerofilepath = Path("Zero_HAND_Frames/frame" + str(frameCounter) + ".json")
                Onefilepath = Path("One_HAND_Frames/frame" + str(frameCounter) + ".json")
                Zerodataframe = pd.DataFrame(list(results.multi_hand_world_landmarks[0].landmark))
                Onedataframe = pd.DataFrame(list(results.multi_hand_world_landmarks[1].landmark))
                ZeroDF = Zerodataframe.to_json(Zerofilepath, indent=0)
                OneDF = Onedataframe.to_json(Onefilepath, indent=0)
                frameCounter += 1
        else:
            continue
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
# # DATA ORGANISATION SECTION

frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# hardcoded atm because am lazy to implement ffmpeg/moviepy etc, etc into this
frame_number = 5560
header = ["landmark", "x", "y", "z", "distance"]
with open("Zero_HAND_cords_tracker.csv", "w", newline="") as ZeroHand:
    writer = csv.writer(ZeroHand)
    writer.writerow(head for head in header)

with open("One_HAND_cords_tracker.csv", "w", newline="") as OneHand:
    writer = csv.writer(OneHand)
    writer.writerow(head for head in header)

# open the json frames on the background.
for frames in range(frame_number):
    with open("Zero_HAND_Frames/frame" + str(frame_counter) + ".json", "r") as f:
        data = json.load(f)
        frame_counter += 1
    # track the landmarks and write to csv file.
    numLandmark = 0
    while numLandmark < 22:
        for landmark in data["0"][str(numLandmark)], :
            # takes the relevant information from the JSON files.
            x = float(landmark["x"])
            y = float(landmark["y"])
            z = float(landmark["z"])
            dist = x*x + y*y
            distance = dist*dist
            coords = numLandmark, x, y, z, distance
            # opens the csv file, writes a row, then stops writing when landmark threshold is reached; 33.
            with open("Zero_HAND_cords_tracker.csv", "a", newline="") as cordF:
                writer = csv.writer(cordF)
                writer.writerow(coords)
            numLandmark += 1
    # just to aesthetically show frame count on the run window.
    print("Zero-Hand saved frame: "+str(frame_counter))

frame_counter = 0
frame_number = 5560
# open the json frames on the background.
for frames in range(frame_number):
    with open("One_HAND_Frames/frame" + str(frame_counter) + ".json", "r") as f:
        data = json.load(f)
        frame_counter += 1
    # track the landmarks and write to csv file.
    numLandmark = 0
    while numLandmark < 22:
        for landmark in data["0"][str(numLandmark)], :
            # takes the relevant information from the JSON files.
            x = float(landmark["x"])
            y = float(landmark["y"])
            z = float(landmark["z"])
            dist = x*x + y*y
            distance = dist*dist
            coords = numLandmark, x, y, z, distance
            # opens the csv file, writes a row, then stops writing when landmark threshold is reached; 22.
            with open("One_HAND_cords_tracker.csv", "a", newline="") as cordF:
                writer = csv.writer(cordF)
                writer.writerow(coords)
            numLandmark += 1
    # just to aesthetically show frame count on the run window.
    print("One-Hand saved frame: "+str(frame_counter))

with open("Zero_cords_tracker.csv") as data:
    df = pd.read_csv(data)
    sort = df.sort_values(["landmark"])
    sort.to_csv("Zero_HAND_sorted_landmark.csv", index=False)

with open("One_cords_tracker.csv") as data:
    df = pd.read_csv(data)
    sort = df.sort_values(["landmark"])
    sort.to_csv("One_HAND_sorted_landmark.csv", index=False)

sig_list = []
# opens the sorted file and extracts the landmark and distance information.
curLM = 0
with open("Zero_sorted_landmark.csv") as tbPlot:
    df = pd.read_csv(tbPlot, usecols=["landmark", "distance"])
    # for each landmark creates a CSV file.
    while curLM < 22:
        lm = df.query("landmark == "+str(curLM))
        lmFilePath = Path("Zero_HAND_Landmark_CSVs/Landmark" + str(curLM) + ".csv")
        lmDF = lm.to_csv(lmFilePath, index=False)
        curLM += 1

sig_list = []
# opens the sorted file and extracts the landmark and distance information.
curLM = 0
with open("One_sorted_landmark.csv") as tbPlot:
    df = pd.read_csv(tbPlot, usecols=["landmark", "distance"])
    # for each landmark creates a CSV file.
    while curLM < 22:
        lm = df.query("landmark == "+str(curLM))
        lmFilePath = Path("One_HAND_Landmark_CSVs/Landmark" + str(curLM) + ".csv")
        lmDF = lm.to_csv(lmFilePath, index=False)
        curLM += 1


# From the create CSV files takes distance information.
fileLM = 0
while fileLM < 22:
    with open("Zero_Landmark_CSVs/Landmark"+str(fileLM)+".csv") as f:
        testdf = pd.read_csv(f, usecols=["distance"])
        # creates a list of the distance data.
        new_list = list(testdf.distance)
        # takes the mean score of the distance data.
        dist_mean = float(testdf.mean(axis=0))
        sig_list = []
    # reads the landmarks again for pandas reasons.
    with open("Zero_Landmark_CSVs/Landmark"+str(fileLM)+".csv") as g:
        sigDF = pd.read_csv(g)
        # conditionally checks the distance information and compares with the mean score.
        # if bigger than mean, that means a significant movement has occurred.
        for dist in new_list:
            if dist < dist_mean:
                sig_list.append(0)
            else:
                sig_list.append(1)
        sigDF['significance'] = sig_list
        sigDF.append(sig_list)
        # writes back the significance values onto the original landmark data.
        sigDF.to_csv("Zero_Landmark_CSVs/Zero_HAND_Landmark"+str(fileLM)+".csv", index=False)
        fileLM += 1

# From the create CSV files takes distance information.
fileLM = 0
while fileLM < 22:
    with open("One_Landmark_CSVs/Landmark"+str(fileLM)+".csv") as f:
        testdf = pd.read_csv(f, usecols=["distance"])
        # creates a list of the distance data.
        new_list = list(testdf.distance)
        # takes the mean score of the distance data.
        dist_mean = float(testdf.mean(axis=0))
        sig_list = []
    # reads the landmarks again for pandas reasons.
    with open("One_Landmark_CSVs/Landmark"+str(fileLM)+".csv") as g:
        sigDF = pd.read_csv(g)
        # conditionally checks the distance information and compares with the mean score.
        # if bigger than mean, that means a significant movement has occurred.
        for dist in new_list:
            if dist < dist_mean:
                sig_list.append(0)
            else:
                sig_list.append(1)
        sigDF['significance'] = sig_list
        sigDF.append(sig_list)
        # writes back the significance values onto the original landmark data.
        sigDF.to_csv("One_Landmark_CSVs/Zero_HAND_Landmark"+str(fileLM)+".csv", index=False)
        fileLM += 1
