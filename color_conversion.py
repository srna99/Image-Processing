import numpy as np


def convert_single_spectrum(img, color):
    img_single_spec = np.round(np.mean(img, axis=2))

    if color.lower() == 'gray':
        return np.uint8(img_single_spec)
    elif color.lower() == 'red':
        red_img = np.uint8([[val, 0, 0] for row in img_single_spec for val in row])
        red_img.shape = img_single_spec.shape[0], img_single_spec.shape[1], 3
        return red_img
    elif color.lower() == 'green':
        green_img = np.uint8([[0, val, 0] for row in img_single_spec for val in row])
        green_img.shape = img_single_spec.shape[0], img_single_spec.shape[1], 3
        return green_img
    elif color.lower() == 'blue':
        blue_img = np.uint8([[0, 0, val] for row in img_single_spec for val in row])
        blue_img.shape = img_single_spec.shape[0], img_single_spec.shape[1], 3
        return blue_img
