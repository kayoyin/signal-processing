import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def show(transform, data, output=None):
    '''
        size of transform : (n, 2)
        size of data : (n, 784)
        '''
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(transform[:, 0], transform[:, 1])
    for x0, y0, img in zip(transform[:, 0], transform[:, 1], data.reshape((-1, 28, 28), order='F')):
        ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray'), (x0, y0), frameon=False)
        ax.add_artist(ab)

    if output is None:
        plt.show()
    else:
        plt.savefig(output)

def mean_square_error(A, B):
    """
    Compute the mean square error between two images
    :param A: a numpy array of shape (img_height, img_width)
    :param B: a numpy array of shape (img_height, img_width)
    :return: a scalar
    """
    return ((A - B)**2).mean()

def distance_mat(X, n_neighbors=6):
    """
    Compute the square distance matrix using Euclidean distance
    :param X: Input data, a numpy array of shape (img_height, img_width)
    :param n_neighbors: Number of nearest neighbors to consider, int
    :return: numpy array of shape (img_height, img_height), numpy array of shape (img_height, n_neighbors)
    """
    def dist(a, b):
        return np.sqrt(sum((a - b)**2))

    # Compute full distance matrix
    distances = np.array([[dist(p1, p2) for p2 in X] for p1 in X])

    # Keep only the 6 nearest neighbors, others set to 0 (= unreachable)
    neighbors = np.zeros_like(distances)
    sort_distances = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]
    for k,i in enumerate(sort_distances):
        neighbors[k,i] = distances[k,i]
    return neighbors, sort_distances

def center(K):
    """
    Method to center the distance matrix
    :param K: numpy array of shape mxm
    :return: numpy array of shape mxm
    """
    n_samples = K.shape[0]

    # Mean for each row/column
    meanrows = np.sum(K, axis=0) / n_samples
    meancols = (np.sum(K, axis=1)/n_samples)[:, np.newaxis]

    # Mean across all rows (entire matrix)
    meanall = meanrows.sum() / n_samples

    K -= meanrows
    K -= meancols
    K += meanall
    return K

