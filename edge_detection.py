import numpy as np

import sub_matrices


def apply_operator(img, op_x, op_y):
    filtered_matrix = []

    mask_size_matrices = sub_matrices.get_mask_size_matrices(img, op_x.shape[0])

    for matrix in mask_size_matrices:
        px_x = np.sum(np.multiply(matrix, op_x))
        px_y = np.sum(np.multiply(matrix, op_y))
        filtered_matrix.append(np.sqrt(px_x ** 2 + px_y ** 2))

    filt_img = np.array(filtered_matrix).reshape(img.shape).astype(np.uint8)

    return filt_img


def prewitt_operator(img):
    op_x = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])
    op_y = np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])

    edge_img = apply_operator(img, op_x, op_y)

    return edge_img


def sobel_operator(img):
    op_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    op_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

    edge_img = apply_operator(img, op_x, op_y)

    return edge_img
