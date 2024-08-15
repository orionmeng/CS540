from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    centered = x - np.mean(x, axis=0)
    return centered.astype(float)

def get_covariance(dataset):
    transposed = np.transpose(dataset)
    covariance = np.dot(transposed, dataset)
    covariance = covariance / (dataset.shape[0] - 1)
    return covariance

def get_eig(S, m):
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    selected_eigenvalues = sorted_eigenvalues[:m]
    selected_eigenvectors = sorted_eigenvectors[:, :m]
    diagonal_matrix = np.diag(selected_eigenvalues)

    return diagonal_matrix, selected_eigenvectors

def get_eig_prop(S, prop):
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    variance = np.sum(sorted_eigenvalues)
    ratio = np.cumsum(sorted_eigenvalues) / variance
    index = np.argmax(ratio >= prop)

    selected_eigenvalues = sorted_eigenvalues[:index+1]
    selected_eigenvectors = sorted_eigenvectors[:, :index+1]
    diagonal_matrix = np.diag(selected_eigenvalues)

    return diagonal_matrix, selected_eigenvectors

def project_image(image, U):
    transposed = np.transpose(U)
    projection = np.dot(U, np.dot(transposed, image))
    return projection

def display_image(orig, proj):
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)

    reshaped_orig = orig.reshape(64, 64)
    reshaped_proj = proj.reshape(64, 64)

    im1 = ax1.imshow(reshaped_orig, aspect='equal')
    ax1.set_title("Original")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(reshaped_proj, aspect='equal')
    ax2.set_title("Projection")
    fig.colorbar(im2, ax=ax2)

    return fig, ax1, ax2
