import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
import json


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
frameCounter = 0
videoFile = "sub-002_ses-002_task-hokey_cokey_run-005_video.mp4"

cap = cv2.VideoCapture(videoFile)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
            filepath = Path("Frame Collector __tester__/frame" + str(frameCounter) + ".json")
            dataframe = pd.DataFrame(list(results.pose_world_landmarks.landmark))
            newDF = dataframe.to_json(filepath, indent=1)
            frameCounter += 1
            for landmark in ("Frame Collector/frame" + str(frameCounter) + ".json", "r")["0"]["0"], :
                df1 = float(landmark["x"]), float(landmark["y"]), float(landmark["z"])
                write_to_excel = df1.to_excel("Coordinate_Holder.xlsx")
        else:
            continue
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


# def sklearn_haversine(movement):
#     haversine = DistanceMetric.get_metric('haversine')
#     movement = np.hstack((movement[]))
#     dists = haversine.pairwise(movement)
#     return 6371 * dists

# for frames in int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):

# filepath = Path("Frame Collector/frame" + str(frameCounter) + ".json")
# data = json.dumps(results.pose_world_landmarks.landmark)
# newDF = data.to_json(filepath, indent=1)