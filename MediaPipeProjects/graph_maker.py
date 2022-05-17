import matplotlib.pyplot as plt
import pandas as pd

with open("FULLBODY_Data/FB_Landmark_CSVs/FB_Landmark5.csv") as f:
    df = pd.read_csv(f)

xpoints = df.significance
ypoints = df.distance

plt.plot(xpoints, ypoints, ls=":")
plt.title("n y o o m ! !")
plt.ylabel("significance")
plt.xlabel("frames")
plt.show()

