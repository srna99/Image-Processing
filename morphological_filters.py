import numpy as np

import sub_matrices


def dilation(img, se_size):
    structuring_element = np.full((se_size, se_size), 255)

    morphed_matrix = []

    mask_size_matrices = sub_matrices.get_mask_size_matrices(img, se_size)

    for matrix in mask_size_matrices:
        morphed_matrix.append(255 if np.any(matrix == structuring_element) else 0)

    dil_img = np.array(morphed_matrix).reshape(img.shape).astype(np.uint8)

    return dil_img


def erosion(img, se_size):
    structuring_element = np.full((se_size, se_size), 255)

    morphed_matrix = []

    mask_size_matrices = sub_matrices.get_mask_size_matrices(img, se_size)

    for matrix in mask_size_matrices:
        morphed_matrix.append(255 if np.all(matrix == structuring_element) else 0)

    ero_img = np.array(morphed_matrix).reshape(img.shape).astype(np.uint8)

    return ero_img
