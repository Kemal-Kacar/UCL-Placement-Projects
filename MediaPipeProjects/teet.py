import matplotlib.pyplot as plt
import pandas as pd
lm = []
x = 0
with open("sorted_landmark.csv") as tbPlot:
    df = pd.read_csv(tbPlot, usecols=["landmark", "distance"])
    current_landmark = 0
    for data in df.distance:
        while df.landmark == current_landmark:
            lm.append(df.distance)

plt.plot(x, lm, linestyle=":", color="gold", linewidth=1)
plt.title("Distance within landmarks")
plt.ylabel("distance")
plt.xlabel("landmark")
plt.show()

