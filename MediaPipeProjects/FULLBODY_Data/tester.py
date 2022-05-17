import pandas as pd
import mediapipe as mp
import cv2
import time

videoFile = "FB_Video_data/IMG_5852.MOV"
# mediapipe stuff, dependencies.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
frameCounter = 0
# change the () area to 0 if you want to use actual camera (in order of computer camera).
cap = cv2.VideoCapture(videoFile)
start = time.time()
print("start of process!")
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
            filepath = Path("FB_Frames/frame" + str(frameCounter) + ".json")
            dataframe = pd.DataFrame(list(results.pose_world_landmarks.landmark))
            newDF = dataframe.to_json(filepath, indent=0)
            frameCounter += 1
        else:
            continue
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

def extractor(x, y, z):
    with open("random.csv", "a") as extractorData:
        for LMN in range(results.landmark):
            df = pd.DataFrame(extractorData)
            df[x] = [results.pose_world_landmarks.landmark["x"]]
            df[y] = [results.pose_world_landmarks.landmark["y"]]
            df[z] = [results.pose_world_landmarks.landmark["z"]]
            return df


extractor([results.pose_world_landmarks.landmark["x"]],
          [results.pose_world_landmarks.landmark["y"]],
          [results.pose_world_landmarks.landmark["z"]])