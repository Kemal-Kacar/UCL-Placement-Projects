import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


# def re_shaper(frames, data_array):
#     frame = 000
#     for i in range(frames):
#         while frame < frames:
#             # data_array1 = data_array[1:]
#             lm_info = np.reshape(data_array[frame][1:], (478, 3))
#             frame += 1
#             print(lm_info)


def animate(scat_plot):
    y_i = F[i, ::3]
    scat.set_offsets(np.c_[x[::3], y_i])


with open("_18_05_2022-14_26_32_loc.dat", "rb") as f:
    data = np.fromfile(f, "float32")
    data2 = np.reshape(data, (511, 1435))
    frames = 511
    frame = 000
    .set(xlim=(-3, 3), ylim=(-1, 1))
    for i in range(frames):
        plt.figure(figsize=(5, 5))

        if frame < frames:
            # data_array1 = data_array[1:]
            lm_info = np.reshape(data2[frame][1:], (478, 3))
            landmark = 0
            for lm in range(478):
                x = np.take(-lm_info[landmark], 0)
                y = np.take(-lm_info[landmark], 1)
                landmark += 1
                plt.scatter(x, y)
            frame += 1
            plt.show()

    # for array in data2:

z = 9
# reshape the array to accommodate the individual coordinates.
# plt scatterplot for the coordinates, 3d scatterplot
# then get all those landmarks and plot them in a for loop for the scatterplot datapoints.