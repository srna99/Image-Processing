import os
import numpy as np


training_set = []
total_pixels = 0


def extract_features(hist, label):
    bin_hist = np.asarray([hist[0], hist[255]])

    global total_pixels
    total_pixels = np.sum(bin_hist)

    features = [calculate_area(bin_hist),
                calculate_mean(bin_hist),
                calculate_standard_deviation(bin_hist),
                calculate_entropy(bin_hist),
                label]

    training_set.append(features)


def calculate_area(hist):
    return hist[1]


def calculate_mean(hist):
    pixel_sum = hist[1] * 255

    mean = pixel_sum / total_pixels

    return round(mean, 4)


def calculate_standard_deviation(hist):
    mean = calculate_mean(hist)
    pixel_sum = 0

    for i in range(2):
        pixel_val = i * 255
        pixel_sum += ((pixel_val - mean) * (pixel_val - mean)) * hist[i]

    variance = pixel_sum / total_pixels
    std = np.math.sqrt(variance)

    return round(std, 4)


def calculate_entropy(hist):
    norm_hist = hist / total_pixels

    entropy = -np.sum(np.multiply(norm_hist, np.log(norm_hist)))

    return round(entropy, 4)


def save_training_set():
    dataset = np.array(training_set)
    np.savetxt(os.path.join('output', 'data.csv'), dataset, delimiter=',', fmt='%s')
