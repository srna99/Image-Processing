import numpy as np


def add_salt_pepper_noise(img, strength):
    num_pixels = img.shape[0] * img.shape[1]
    sp_img = img.copy()

    num_salt = np.ceil(strength * num_pixels * 0.5)

    for i in range(int(num_salt)):
        row = np.random.randint(0, img.shape[0] - 1)
        col = np.random.randint(0, img.shape[1] - 1)
        sp_img[row, col] = 255

    num_pepper = np.ceil(strength * num_pixels * 0.5)

    for i in range(int(num_pepper)):
        row = np.random.randint(0, img.shape[0] - 1)
        col = np.random.randint(0, img.shape[1] - 1)
        sp_img[row, col] = 0

    return sp_img


def add_gaussian_noise(img, mean, std):
    noise = np.random.normal(mean, std, (img.shape[0], img.shape[1]))
    gauss_img = np.uint8(img + noise)

    return gauss_img
