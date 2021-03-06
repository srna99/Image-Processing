import os
import numpy as np
import matplotlib.pyplot as plt


class_hist = {}
class_counts = {}
num_pixels = 0


def generate_histogram(img):
    global num_pixels
    num_pixels = img.shape[0] * img.shape[1]

    if img.ndim == 2:
        hist = np.zeros(256)

        for l in range(256):
            hist[l] = np.sum(np.sum(img == l))
    else:
        hist = np.zeros((256, 3))

        for channel in range(3):
            for l in range(0, 256):
                hist[l, channel] = np.sum(np.sum(img[:, :, channel] == l))

    return hist


def average_histogram(img, img_class):
    hist = generate_histogram(img)

    if img_class not in class_hist.keys():
        class_hist[img_class] = hist
        class_counts[img_class] = 1
    else:
        class_hist[img_class] += hist
        class_counts[img_class] += 1

    avg_hist = class_hist[img_class] / class_counts[img_class]

    return avg_hist


def equalize_histogram(img):
    hist = generate_histogram(img)

    norm_hist = hist / num_pixels
    equ_hist = np.floor(255 * np.cumsum(norm_hist)).astype(np.uint8)

    equ_img = equ_hist[img]
    equ_img = equ_img.reshape(img.shape)

    return equ_img


def quantize_histogram(img, levels):
    delta = 256 / levels
    quan_img = delta * np.floor(img / delta) + delta / 2
    quan_img = np.reshape(quan_img, img.shape).astype(np.uint8)

    msqe = calculate_msqe(img, quan_img)

    return quan_img, msqe


def segment_histogram(img):
    hist = generate_histogram(img)

    norm_hist = hist / num_pixels
    bins = np.arange(256)

    threshold = 0
    min_within_group_variance = float('inf')

    for l in range(256):
        obj_prior_prob = np.sum(norm_hist[:l])
        bg_prior_prob = np.sum(norm_hist[l:])

        if obj_prior_prob == 0 or bg_prior_prob == 0:
            continue

        obj_mean = np.sum(np.multiply(norm_hist[:l], bins[:l])) / obj_prior_prob
        bg_mean = np.sum(np.multiply(norm_hist[l:], bins[l:])) / bg_prior_prob

        obj_variance = np.sum(((bins[:l] - obj_mean) ** 2) * norm_hist[:l]) / obj_prior_prob
        bg_variance = np.sum(((bins[l:] - bg_mean) ** 2) * norm_hist[l:]) / bg_prior_prob

        within_group_variance = (obj_variance ** 2) * obj_prior_prob + (bg_variance ** 2) * bg_prior_prob

        if within_group_variance < min_within_group_variance:
            min_within_group_variance = within_group_variance
            threshold = l

    seg_img = convert_binary_image(img, threshold)

    return seg_img


def save_histogram(hist, img_name, mode):
    plt.clf()

    plt.title('Histogram of ' + img_name)
    plt.xlabel('l')
    plt.ylabel('h(l)')

    if hist.ndim == 1:
        plt.ylim([0, np.amax(hist) * 1.1])
        plt.bar(np.arange(256), hist, color='black')
    else:
        set_ylim = False

        if not(np.amax(hist) == num_pixels):
            plt.ylim([0, np.amax(hist) * 1.1])
            set_ylim = True

        for channel, color in zip(range(3), ['red', 'green', 'blue']):
            if hist[0, channel] == num_pixels:
                continue

            if not set_ylim:
                plt.ylim([0, np.amax(hist[:, channel]) * 1.1])

            plt.bar(np.arange(256), hist[:, channel], color=color, alpha=0.4)

    plt.savefig(os.path.join('output', img_name + '_' + mode + '.jpg'))
    plt.close()


def calculate_msqe(img, quan_img):
    return np.mean(np.square(img - quan_img))


def convert_binary_image(img, threshold):
    bin_img = np.where(img <= threshold, 255, img)
    final_bin_img = np.where(img > threshold, 0, bin_img)

    return final_bin_img
