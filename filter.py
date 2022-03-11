import numpy as np


def get_mask_size_matrices(img, mask_size):
    rows, cols = img.shape
    pad_img = np.pad(img, mask_size - 2, mode='edge')

    matrix_of_matrices = []
    for i in range(0, rows):
        for j in range(0, cols):
            matrix = []

            for x in range(i, i + mask_size):
                values = []

                for y in range(j, j + mask_size):
                    values.append(pad_img[x][y])

                matrix.append(values)

            matrix_of_matrices.append(matrix)

    return np.array(matrix_of_matrices)


def apply_linear_filter(img, mask_size, weights):
    filtered_matrix = []

    mask_size_matrices = get_mask_size_matrices(img, mask_size)

    for matrix in mask_size_matrices:
        new_pixel = np.multiply(matrix, weights)
        filtered_matrix.append(np.sum(new_pixel))

    filt_img = np.array(filtered_matrix).reshape(img.shape).astype(np.uint8)

    return filt_img


def apply_median_filter(img, mask_size, weights):
    filtered_matrix = []

    mask_size_matrices = get_mask_size_matrices(img, mask_size)

    for matrix in mask_size_matrices:
        values = []

        for pix, w in zip(matrix.reshape(-1), weights.reshape(-1).astype(np.uint8)):
            values.extend([pix] * w)

        values.sort()
        median = values[len(values) // 2]

        filtered_matrix.append(median)

    filt_img = np.array(filtered_matrix).reshape(img.shape).astype(np.uint8)

    return filt_img
