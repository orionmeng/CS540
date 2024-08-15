from scipy.linalg import eigh
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    data = []
    with open(filepath, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

def calc_features(row):
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    return np.array([x1, x2, x3, x4, x5, x6], dtype = np.float64)

def hac(features):
    Z = []
    n = len(features)
    m = len(features)   # H[m - n] for adding new cluster
    H = np.ones(n-1)    # size of each additional cluster

    cluster_indices = list(range(n))
    distance_matrix = np.zeros((n, n))

    def euclidean_distance(x, y):
        return np.linalg.norm(x - y)

    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = distance_matrix[j, i] = euclidean_distance(features[i], features[j])

    while len(cluster_indices) > 1:
        print(distance_matrix)
        min_distance = float('inf')
        index_1 = None
        index_2 = None

        for i in range(len(cluster_indices)):
            for j in range(i+1, len(cluster_indices)):
                distance = distance_matrix[i, j]
                if distance < min_distance:
                    min_distance = distance
                    index_1 = i
                    index_2 = j
                elif distance == min_distance:
                    if cluster_indices[i] < cluster_indices[index_1]:
                        index_1 = i
                        index_2 = j
                    elif cluster_indices[i] == cluster_indices[index_1]:
                        if cluster_indices[j] < cluster_indices[index_2]:
                            index_1 = i
                            index_2 = j

        linkage_distance = distance_matrix[index_1, index_2]
        cluster_1 = cluster_indices[index_1]
        if cluster_1 < n:
            size_1 = 1
        else:
            size_1 = H[cluster_1 - n]
        cluster_2 = cluster_indices[index_2]
        if cluster_2 < n:
            size_2 = 1
        else:
            size_2 = H[cluster_2 - n]
        H[m - n] = size_1 + size_2
        Z.append([np.round(cluster_1), np.round(cluster_2), linkage_distance, np.round(H[m - n])])

        new_cluster_indices = [cluster_indices[i] for i in range(len(cluster_indices)) if i != index_1 and i != index_2]
        new_cluster_indices.append(m)
        m += 1

        new_distance_matrix = np.zeros((len(new_cluster_indices), len(new_cluster_indices)))
        indices_to_skip = [index_1, index_2]
        indices_to_keep = [idx for idx in range(len(distance_matrix)) if idx not in indices_to_skip]
        for i, new_i in enumerate(indices_to_keep):
            for j, new_j in enumerate(indices_to_keep):
                new_distance_matrix[i][j] = distance_matrix[new_i][new_j]
        for i, new_i in enumerate(indices_to_keep):
            complete_distance = max(
                distance_matrix[new_i, index_1],
                distance_matrix[new_i, index_2]
            )
            new_distance_matrix[i][-1] = complete_distance
            new_distance_matrix[-1][i] = complete_distance

        cluster_indices = new_cluster_indices
        distance_matrix = new_distance_matrix

    return np.array(Z)

def fig_hac(Z, names):
    fig = plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig

def normalize_features(features):
    features_array = np.array(features)
    means = np.mean(features_array, axis=0)
    stds = np.std(features_array, axis=0)
    normalized_features = []
    for feature_vector in features_array:
        normalized_feature_vector = (feature_vector - means) / stds
        normalized_features.append(normalized_feature_vector)
    return normalized_features
