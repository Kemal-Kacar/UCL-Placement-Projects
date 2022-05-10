import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
import csv
import json

# all in one big package that has everything, mo-cap to frame data, to organising all data, to b a r s i n g.

# mediapipe stuff, dependencies.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
frameCounter = 0
# the video file. Just change the "" area.
videoFile = "sub-002_ses-002_task-hokey_cokey_run-005_video.mp4"

# change the () area to 0 if you want to use actual camera (in order of computer camera).
cap = cv2.VideoCapture(videoFile)
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
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
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # place to put the extracted data.
        if frameCounter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            filepath = Path("Frame Collector/frame" + str(frameCounter) + ".json")
            dataframe = pd.DataFrame(list(results.pose_world_landmarks.landmark))
            newDF = dataframe.to_json(filepath, indent=1)
            frameCounter += 1
        else:
            continue
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# currently hardcoded, can get from the file properties, am lazy atm.
frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
header = ["landmark", "x", "y", "z", "distance"]
with open("cords_tracker.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(head for head in header)

# open the json frames on the background.
for frames in range(frame_number):
    with open("Frame Collector/frame" + str(frame_counter) + ".json", "r") as f:
        data = json.load(f)
        frame_counter += 1
    # track the landmarks and write to csv file.
    numLandmark = 0
    while numLandmark < 33:
        for landmark in data["0"][str(numLandmark)], :
            # takes the relevant information from the JSON files.
            x = float(landmark["x"])
            y = float(landmark["y"])
            z = float(landmark["z"])
            dist = x*x + y*y
            distance = dist*dist
            coords = numLandmark, x, y, z, distance
            # opens the csv file, writes a row, then stops writing when landmark threshold is reached; 33.
            with open("cords_tracker.csv", "a", newline="") as cordF:
                writer = csv.writer(cordF)
                writer.writerow(coords)
            numLandmark += 1
    # just to aesthetically show frame count on the run window.
    print("saved frame: "+str(frame_counter))

with open("cords_tracker.csv") as data:
    df = pd.read_csv(data)
    sort = df.sort_values(["landmark"])
    sort.to_csv("sorted_landmark.csv", index=False)

sig_list = []
# opens the sorted file and extracts the landmark and distance information.
curLM = 0
with open("sorted_landmark.csv") as tbPlot:
    df = pd.read_csv(tbPlot, usecols=["landmark", "distance"])
    # for each landmark creates a CSV file.
    while curLM < 33:
        lm = df.query("landmark == "+str(curLM))
        lmFilePath = Path("Landmark_CSVs/Landmark" + str(curLM) + ".csv")
        lmDF = lm.to_csv(lmFilePath, index=False)
        curLM += 1

# From the create CSV files takes distance information.
fileLM = 0
while fileLM < 33:
    with open("Landmark_CSVs/Landmark"+str(fileLM)+".csv") as f:
        testdf = pd.read_csv(f, usecols=["distance"])
        # creates a list of the distance data.
        new_list = list(testdf.distance)
        # takes the mean score of the distance data.
        dist_mean = float(testdf.mean(axis=0))
        sig_list = []
    # reads the landmarks again for pandas reasons.
    with open("Landmark_CSVs/Landmark"+str(fileLM)+".csv") as g:
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
        sigDF.to_csv("Landmark_CSVs/Landmark"+str(fileLM)+".csv", index=False)
        fileLM += 1