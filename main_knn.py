import sys
import numpy as np

import knn


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        dataset_path = f.readline().strip()
        k = int(f.readline().strip())

    dataset = np.loadtxt(dataset_path, delimiter=',')

    accuracy = knn.cross_validation_knn(dataset, k)

    print('Accuracy of KNN with {} neighbors: {:.2f}%'.format(k, accuracy * 100))
