import numpy as np


def cross_validation_knn(dataset: np.array, k):
    splits_indices = get_cross_validation_splits(dataset)
    print(splits_indices)

    sum_of_accuracies = 0

    for fold in range(10):
        fold_indices = splits_indices[fold]

        if fold == 0:
            train = dataset[fold_indices[1] :]
        elif fold == 9:
            train = dataset[: fold_indices[0]]
        else:
            train = np.concatenate((dataset[: fold_indices[0]], dataset[fold_indices[1] :]))

        print(train)

        test = dataset[fold_indices[0] : fold_indices[1]]
        test_data, test_labels = test[:, : -1], test[:, -1]

        print(test)

        # sum_of_accuracies += knn(train, test_data, test_labels, k)

    # return round(sum_of_accuracies / 10, 4)


def get_cross_validation_splits(dataset):
    size = dataset.shape[0]
    size_of_splits = size // 10

    splits_indices = []
    index = 0

    for fold in range(10):
        if fold == 9:
            splits_indices.append([index, size])
        else:
            end_index = index + size_of_splits
            splits_indices.append([index, end_index])
            index = end_index

    return splits_indices


def knn(train, test_data, test_labels, k):
    predictions = np.full(test_data.shape[0], -1)

    for test_inst in test_data:
        candidates = [float('inf')] * k
        # class

        for train_inst in train:
            dist = calculate_distance(test_inst, train_inst)

            for idx, cand in enumerate(candidates):
                # if dist < cand[0]:

    return calculate_accuracy(test_labels, predictions)


def calculate_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def calculate_accuracy(actual, predicted):
    return np.sum(actual == predicted) / actual.shape[0]
