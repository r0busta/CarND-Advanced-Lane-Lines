import matplotlib.pyplot as plt
import math


def plot_images(images):
    fig = plt.figure(figsize=(15, 10))
    ncols = 5 if len(images) >= 5 else len(images)
    nrows = math.ceil(len(images) / ncols)
    for i in range(len(images)):
        fig.add_subplot(nrows, ncols, i+1)
        img = images[i]
        plt.imshow(img, cmap='gray')

    plt.show()


def moving_avg(curr_avg, val, samples_count):
    res = curr_avg
    res -= res / samples_count
    res += val / samples_count
    return res
