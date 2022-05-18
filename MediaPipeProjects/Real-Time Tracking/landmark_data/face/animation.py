import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
time = np.array(0, ndmin=1)
lm1 = np.array(0, ndmin=3)


with open("_18_05_2022-14_26_32_loc.dat", "rb") as f:
    data = np.fromfile(f, "float32")
    data2 = np.reshape(data, (511, 1435))
    for frame in data2:
        np.append(frame[0], time)
        np.append(frame[1], lm1)

z = 9
