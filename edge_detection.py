import numpy as np

import sub_matrices


def apply_operator(img, operator):
    filtered_matrix = []

    mask_size_matrices = sub_matrices.get_mask_size_matrices(img, operator.shape[0])

    for matrix in mask_size_matrices:
        px_x = np.sum(np.multiply(matrix, operator))
        px_y = np.sum(np.multiply(matrix, operator.T))
        filtered_matrix.append(np.sqrt(px_x ** 2 + px_y ** 2))

    filt_img = np.array(filtered_matrix).reshape(img.shape).astype(np.uint8)

    return filt_img


def prewitt_operator(img):
    prewitt = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])

    edge_img = apply_operator(img, prewitt)

    return edge_img


def sobel_operator(img):
    sobel = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

    edge_img = apply_operator(img, sobel)

    return edge_img
