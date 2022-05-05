import matplotlib.pyplot as plt
import pandas as pd

with open("Landmark_CSVs/Landmark0.csv") as f:
    df = pd.read_csv(f)
    print(df.significance)

sig_or_nah = [0, 1]

plt.plot(df.significance, sig_or_nah, ":", label="yeet")
plt.title("n y o o m ! !")
plt.ylabel("significance")
plt.xlabel("frames")
plt.show()

