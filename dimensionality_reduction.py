import glob
import cv2
import numpy as np
from utils import *

def pca(img, n_components=9):
    """
    Perform PCA dimensionality reduction on the input image
    :param img: a numpy array of shape (n_samples, dim_images)
    :param n_components: number of principal components for projection
    :return: image in PCA projection, a numpy array of shape (n_samples, n_components)
    """

    # Compute the covariance matrix
    cov_mat = np.cov(img.T)

    # Compute the eigenvectors and eigenvalues
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [
        (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))
    ]

    # Select n_components eigenvectors with largest eigenvalues, obtain subspace transform matrix
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_pairs = np.array(eig_pairs)
    matrix_w = np.hstack(
        [eig_pairs[i, 1].reshape(img.shape[1], 1) for i in range(n_components)]
    )

    # Return samples in new subspace
    return np.dot(img, matrix_w), matrix_w

def inverse_pca(img, components):
    """
    Obtain the reconstruction after PCA dimensionality reduction
    :param img: Reduced image, a numpy array of shape (n_samples, n_components)
    :param components: a numpy array of size (original_dimension, n_components)
    :return:
    """
    reconstruct = np.dot(img, components.T).astype(int)
    return reconstruct.reshape(-1,28,28)


def mds(data, n_components=2):
    """
    Apply multidimensional scaling (aka Principal Coordinates Analysis)
    :param data: nxn square distance matrix
    :param n_components: number of components for projection
    :return: projected output of shape (n_components, n)
    """

    # Center distance matrix
    center(data)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_val_cov, eig_vec_cov = np.linalg.eig(data)
    eig_pairs = [
        (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))
    ]

    # Select n_components eigenvectors with largest eigenvalues, obtain subspace transform matrix
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_pairs = np.array(eig_pairs)
    matrix_w = np.hstack(
        [eig_pairs[i, 1].reshape(data.shape[1], 1) for i in range(n_components)]
    )

    # Return samples in new subspace
    return matrix_w

def isomap(data, n_components=2, n_neighbors=6, dist=False):
    """
    Dimensionality reduction with isomap algorithm
    :param data: input image matrix of shape (n,m) if dist=False, square distance matrix of size (n,n) if dist=True
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for distance matrix computation
    :param dist: boolean indicating the data type
    :return: Projected output of shape (n_components, n)
    """
    if not dist:
        # Compute distance matrix
        data, _ = distance_mat(data, n_neighbors)

    # Compute shortest paths from distance matrix
    from sklearn.utils.graph import graph_shortest_path
    graph = graph_shortest_path(data, directed=False)
    graph = -0.5 * (graph ** 2)

    # Return the MDS projection on the shortest paths graph
    return mds(graph, n_components)

def LLE(data, neighbors_idx=None, n_components=2, n_neighbors=6):
    """
    Dimensionality reduction with FastLLE algorithm
    :param data: input image matrix of shape (n,m)
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for the weight extraction
    :return: Projected output of shape (n_components, n)
    """
    if neighbors_idx is None:
        # Compute the nearest neighbors
        _, neighbors_idx = distance_mat(data, n_neighbors)

    n = data.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        # Center the neighbors matrix
        k_indexes = neighbors_idx[i, :]
        neighbors = data[k_indexes, :] - data[i, :]

        # Compute the corresponding gram matrix
        gram_inv = np.linalg.pinv(np.dot(neighbors, neighbors.T))

        # Setting the weight values according to the lagrangian
        lambda_par = 2/np.sum(gram_inv)
        w[i, k_indexes] = lambda_par*np.sum(gram_inv, axis=1)/2
    m = np.subtract(np.eye(n), w)
    values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
    return u[:, 1:n_components+1]

def demo():
    images = []
    for path in glob.glob("four_dataset/*.jpg"):
        images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
    flat_images = np.array(images).reshape(982, -1)
    for k in [2, 16, 64, 256]:
        # Perform PCA transform
        pca_images, components = pca(flat_images, k)

        # Check the dimension of the reduced dataset
        print(pca_images.shape)

        # Reconstruct from PCA reduction
        reconstruct = inverse_pca(pca_images, components)

        # Compute MSE between original image and reconstructed image
        mse = [mean_square_error(a,b) for a,b in zip(images, reconstruct)]
        print("Dimension {}, best MSE: {}, worst MSE: {}, MSE over all samples: {}".format(k, min(mse), max(mse), sum(mse)/len(mse)))

        # Save visualization of reductions on four0.jpg
        cv2.imwrite("output/pca_{}.jpg".format(k), reconstruct[0])


if __name__ == "__main__":
    data = np.load("digits-labels.npz")

    # Select columns of five
    d = data['d']
    l = data['l']
    five_idx = np.argwhere(l == 5)
    d_fives = d[:, five_idx].squeeze().T

    # Perform PCA
    transform_pca, components = pca(d_fives, n_components=2)
    print(transform_pca.shape)
    show(transform_pca, d_fives, 'output/pca.png')

    # Perform Isomap
    distances, neighbors_idx = distance_mat(d_fives)
    transform_isomap = isomap(distances, dist=True)
    print(transform_isomap.shape)
    show(transform_isomap, d_fives, 'output/isomap.png')

    # Perform LLE
    transform_lle = LLE(d_fives, neighbors_idx)
    print(transform_lle.shape)
    show(transform_lle, d_fives, 'output/lle.png')
