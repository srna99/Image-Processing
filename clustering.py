import numpy as np


def kmeans(img, k):
    flat_img = img.copy().reshape(-1)

    centroids = np.random.randint(256, size=k)
    clusters = []

    print(centroids)

    iteration = 0
    finished = False

    while iteration < 20 and not finished:
        clusters = assign_clusters(flat_img, centroids)

        centroids, finished = calculate_cluster_means(flat_img, k, centroids, clusters)

        iteration += 1
        print(iteration, centroids)

    km_img = update_img(flat_img, centroids, clusters)
    km_img = km_img.reshape(img.shape)

    return km_img


def assign_clusters(img, centroids):
    clusters = np.zeros(img.shape[0])

    for idx, pixel in enumerate(img):
        min_distance = float('inf')

        for cluster, centroid in enumerate(centroids):
            distance = np.sqrt(((pixel - centroid) ** 2))

            if distance < min_distance:
                min_distance = distance
                clusters[idx] = cluster

    return clusters


def calculate_cluster_means(img, k, centroids, clusters):
    clusters_sums = np.zeros(k)
    cluster_sizes = np.zeros(k)

    for cluster in range(k):
        cluster_points = img[clusters == cluster]

        clusters_sums[cluster] = np.sum(cluster_points)
        cluster_sizes[cluster] = cluster_points.shape[0]

    new_centroids = np.divide(clusters_sums, cluster_sizes).astype(int)

    centroid_diffs = np.abs(centroids - new_centroids)
    finished = np.all(centroid_diffs == 0)

    return new_centroids, finished


def update_img(img, centroids, clusters):
    km_img = img.copy()

    for cluster, centroid in enumerate(centroids):
        clust_img = np.where(clusters == cluster, centroid, km_img)
        km_img = clust_img

    return km_img
