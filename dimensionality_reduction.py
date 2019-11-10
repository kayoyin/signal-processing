import glob
import cv2
import numpy as np

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

def mean_square_error(A,B):
    """
    Compute the mean square error between two images
    :param A: a numpy array of shape (img_height, img_width)
    :param B: a numpy array of shape (img_height, img_width)
    :return: a scalar
    """
    return ((A - B)**2).mean()

def main():
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
    main()