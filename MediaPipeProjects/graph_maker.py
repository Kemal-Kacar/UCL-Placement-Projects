import matplotlib.pyplot as plt
import pandas as pd

with open("Landmark_CSVs/Landmark0.csv") as f:
    df = pd.read_csv(f)


df.plot(x=, y=df.significance, kind="line")
plt.title("n y o o m ! !")
plt.ylabel("significance")
plt.xlabel("frames")
plt.show()

