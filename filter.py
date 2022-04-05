import numpy as np

import sub_matrices


def apply_linear_filter(img, mask_size, weights):
    filtered_matrix = []

    mask_size_matrices = sub_matrices.get_mask_size_matrices(img, mask_size)

    for matrix in mask_size_matrices:
        new_pixel = np.multiply(matrix, weights)
        filtered_matrix.append(np.sum(new_pixel))

    filt_img = np.array(filtered_matrix).reshape(img.shape).astype(np.uint8)

    return filt_img


def apply_median_filter(img, mask_size, weights):
    filtered_matrix = []

    mask_size_matrices = sub_matrices.get_mask_size_matrices(img, mask_size)

    for matrix in mask_size_matrices:
        values = []

        for pix, w in zip(matrix.reshape(-1), weights.reshape(-1).astype(np.uint8)):
            values.extend([pix] * w)

        values.sort()
        median = values[len(values) // 2]

        filtered_matrix.append(median)

    filt_img = np.array(filtered_matrix).reshape(img.shape).astype(np.uint8)

    return filt_img
