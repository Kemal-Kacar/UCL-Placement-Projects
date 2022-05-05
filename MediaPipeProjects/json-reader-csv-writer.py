import json
import csv
import pandas as pd

frame_counter = 0
# currently hardcoded, can get from the file properties, am lazy atm.
frame_number = 5560
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
            distance = x*x + y*y
            coords = numLandmark, x, y, z, distance
            # opens the csv file, writes a row, then stops writing when landmark threshold is reached; 33.
            with open("cords_tracker.csv", "a", newline="") as cordF:
                writer = csv.writer(cordF)
                writer.writerow(coords)
            numLandmark += 1
    # just to aesthetically show framecount on the run window.
    print("saved frame"+str(frame_counter))

with open("cords_tracker.csv") as data:
    df = pd.read_csv(data)
    sort = df.sort_values(["landmark"])
    sort.to_csv("sorted_landmark.csv", index=False)

## Redundant code, I am keeping here just in case...
#             extra code stuff
#             coordDF = pd.DataFrame(coords)
#             coordDF.to_excel('coords_tracker.xlsx', index=False, header=False)
# coordsDF = pd.DataFrame(coords)
# coordsDF.to_csv("cords_tracker.csv", index=False, header=False)

