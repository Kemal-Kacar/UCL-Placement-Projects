# a tester file, dont ask why the name is that way, I am lazy to change it to something else.
import pandas as pd
from pathlib import Path


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


