import numpy as np


def cross_validation_knn(dataset, k):
    splits_indices = get_cross_validation_splits(dataset)

    sum_of_accuracies = 0

    for fold in range(10):
        fold_indices = splits_indices[fold]

        if fold == 0:
            train = dataset[fold_indices[1] :]
        elif fold == 9:
            train = dataset[: fold_indices[0]]
        else:
            train = np.concatenate((dataset[: fold_indices[0]], dataset[fold_indices[1] :]))

        test = dataset[fold_indices[0] : fold_indices[1]]
        test_data, test_labels = test[:, : -1], test[:, -1]

        sum_of_accuracies += knn(train, test_data, test_labels, k)

    return round(sum_of_accuracies / 10, 4)


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

    for t_idx, test_inst in enumerate(test_data):
        candidates = [float('inf')] * k
        cand_classes = [-1] * k

        for train_inst in train:
            dist = calculate_distance(test_inst, train_inst[: -1])

            for c_idx, cand in enumerate(candidates):
                if dist < cand:
                    candidates.insert(c_idx, dist)
                    del candidates[-1]

                    cand_classes.insert(c_idx, train_inst[-1])
                    del cand_classes[-1]

                    break

        class_counts = np.column_stack(np.unique(cand_classes, return_counts=True))

        max_count = np.where(class_counts[:, -1] == np.amax(class_counts[:, -1]))
        equal_classes = class_counts[max_count][:, 0]

        if len(equal_classes) == 1:
            predictions[t_idx] = equal_classes[0]
        else:
            for c in cand_classes:
                if c in equal_classes:
                    predictions[t_idx] = c
                    break

    return calculate_accuracy(test_labels, predictions)


def calculate_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def calculate_accuracy(actual, predicted):
    return np.sum(actual == predicted) / actual.shape[0]
