import numpy as np


def calculate_euclidean_distance(data):

    n_samples = data.shape[0]

    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):

            distance = np.sqrt(
                np.sum((data[i] - data[j]) ** 2)
            )

            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


def calculate_pearson_distance(data):

    n_samples = data.shape[0]

    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):

            x = data[i]
            y = data[j]

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum(
                (x - x_mean) * (y - y_mean)
            )

            denominator = (
                np.sqrt(np.sum((x - x_mean) ** 2))
                * np.sqrt(np.sum((y - y_mean) ** 2))
            )

            if denominator == 0:
                correlation = 0
            else:
                correlation = numerator / denominator

            distance = 1 - correlation

            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix
