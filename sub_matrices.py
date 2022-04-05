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
