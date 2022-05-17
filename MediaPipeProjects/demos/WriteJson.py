import pandas as pd
from path import Path
import mediapipe as mp
from threading import Thread


class ToJSON:
    def __init__(self, directory, landmark, frame):
        self.directory = Path(directory)
        self.landmark = landmark
        self.frame = frame

    def to_jsf(self, dataframe):
        Thread(target=self.landmark, args=()).start()
        dataframe = pd.DataFrame(list(self.landmark))
        dataframe = dataframe.to_json(self.directory, "Frame"+self.frame)

