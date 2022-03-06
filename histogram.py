import os

import numpy as np
import matplotlib.pyplot as plt


avg_hist = {}
class_counts = {}


def generate_histogram(img):
    hist = np.zeros((255, 1))

    for l in range(0, 255):
        hist[l] = np.sum(np.sum(img == l))

    return hist


def average_histogram(img, img_class):
    hist = generate_histogram(img)

    if img_class not in avg_hist.keys():
        avg_hist[img_class] = hist
        class_counts[img_class] = 1
    else:
        avg_hist[img_class] += hist
        class_counts[img_class] += 1

        avg_hist[img_class] /= class_counts[img_class]

    return avg_hist[img_class]


def equalize_histogram(img):
    hist = generate_histogram(img)
    return hist


def save_histogram(hist, img_name, mode):
    plt.clf()

    plt.bar(np.arange(255), np.reshape(hist, -1))
    plt.title('Histogram of ' + img_name)
    plt.xlabel('l')
    plt.ylabel('h(l)')

    plt.savefig(os.path.join('output', img_name + '_' + mode + '.jpg'))
    plt.show()
