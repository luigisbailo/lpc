import math
import sys
import warnings
from math import log, pi

import numpy as np
import scipy
from scipy.linalg import pinv
from scipy.special import digamma, gammaln
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import faiss

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPSILON = 1e-10


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def log_volume_unit_ball_l2(d):
    """
    Compute the logarithm of the volume of a unit L2 ball in d dimensions.
    """
    return (d / 2.0) * log(pi) - gammaln(d / 2.0 + 1.0)


def _compute_kth_neighbor_distances(data, k, dim_threshold, batch_size):
    """
    Compute the k-th nearest neighbor distances for each sample in data.

    Uses scikit-learn's BallTree when the dimension is low and FAISS on GPU
    for high-dimensional data.
    """
    n_samples, n_dimensions = data.shape

    if n_dimensions <= dim_threshold:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
        nbrs.fit(data)
        distances, _ = nbrs.kneighbors(data)
    else:
        # Set up a FAISS GPU index.
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(n_dimensions)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index.add(data)

        distances_list = []
        for i in range(0, n_samples, batch_size):
            batch = data[i: i + batch_size]
            batch_distances, _ = gpu_index.search(batch, k + 1)
            distances_list.append(batch_distances)
        distances = np.vstack(distances_list)

    kth_dists = distances[:, -1]
    return kth_dists


# -----------------------------------------------------------------------------
# Main Metric Functions
# -----------------------------------------------------------------------------
def get_entropy(evaluations, k, base=None, dim_threshold=512, batch_size=128, normalize=False, normalize_by_nodes=True):
    """
    Kozachenko–Leonenko estimator for differential entropy.

    Uses scikit-learn (Ball Tree) for dimensions <= dim_threshold, and batched
    FAISS-GPU otherwise.
    """
    data = evaluations["x_penultimate"].astype(np.float32)
    if normalize:
        data = StandardScaler().fit_transform(data)

    n_samples, n_dimensions = data.shape

    kth_dists = _compute_kth_neighbor_distances(data, k, dim_threshold, batch_size)
    kth_dists += EPSILON  # Avoid log(0)
    avg_log_dist = np.mean(np.log(kth_dists))

    c_d = log_volume_unit_ball_l2(n_dimensions)
    const = digamma(n_samples) - digamma(k) + c_d

    if base is None:
        entropy = const + n_dimensions * avg_log_dist
    else:
        entropy = (const + n_dimensions * avg_log_dist) / math.log(base)

    if normalize_by_nodes:
        entropy /= n_dimensions

    return entropy


def get_coeff_var(evaluations):
    """
    Calculate coefficient of variation metrics of the penultimate layer.
    """
    x_penultimate = evaluations["x_penultimate"]
    norms = np.linalg.norm(x_penultimate, axis=1)
    abs_values = np.abs(x_penultimate)

    norm_mean = np.mean(norms)
    coeff_var_norm = np.std(norms) / norm_mean if norm_mean != 0 else 0.0
    coeff_var_abs = np.std(abs_values) / np.mean(abs_values) if np.mean(abs_values) != 0 else 0.0

    return coeff_var_norm, coeff_var_abs, norm_mean


def get_collapse_metrics(evaluations):
    """
    Calculate collapse metrics in the penultimate layer.

    The metrics include within-class variation, equiangularity, maximum angle,
    equinorm, and within-class covariance.
    """
    x_penultimate = evaluations["x_penultimate"]
    y_predicted = evaluations["y_predicted"]
    y_label = evaluations["y_label"]

    global_mean = np.mean(x_penultimate, axis=0)
    class_mean_centered = []
    sigma_w_list = []
    sigma_b_list = []

    unique_labels = np.unique(y_label)
    for label in unique_labels:
        selection = (y_predicted == label)
        if np.sum(selection) > 0:
            class_mean = np.mean(x_penultimate[selection], axis=0)
            centered_mean = class_mean - global_mean
            class_mean_centered.append(centered_mean)
            sigma_w_list.append(np.cov(x_penultimate[selection].T))
            sigma_b_list.append(np.outer(centered_mean, centered_mean))

    class_mean_centered = np.stack(class_mean_centered, axis=0)
    sigma_w = np.mean(np.stack(sigma_w_list, axis=0), axis=0)
    sigma_b = np.mean(np.stack(sigma_b_list, axis=0), axis=0)
    within_class_variation = np.trace(sigma_w @ pinv(sigma_b)) / len(class_mean_centered)
    within_class_variation_weighted = within_class_variation / x_penultimate.shape[1]
    sigma_w_mean = np.mean(sigma_w)

    n_classes = len(class_mean_centered)
    cosines = []
    cosines_max = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            vec_i = class_mean_centered[i]
            vec_j = class_mean_centered[j]
            norm_product = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
            cosine_sim = np.dot(vec_i, vec_j) / norm_product if norm_product != 0 else 0.0
            cosines.append(cosine_sim)
            cosines_max.append(np.abs(cosine_sim + 1.0 / (n_classes - 1)))

    equiangular = np.std(cosines) if cosines else 0.0
    maxangle = np.mean(cosines_max) if cosines_max else 0.0
    class_norms = np.linalg.norm(class_mean_centered, axis=1)
    equinorm = np.std(class_norms) / np.mean(class_norms) if np.mean(class_norms) != 0 else 0.0

    return {
        "within_class_variation": within_class_variation,
        "within_class_variation_weighted": within_class_variation_weighted,
        "equiangular": equiangular,
        "maxangle": maxangle,
        "equinorm": equinorm,
        "sigma_w": sigma_w_mean,
    }


def get_distance_margins(evaluations):
    """
    Calculate the distance ratio for each sample as a measure of compactness.

    For each sample, the distance ratio is defined as the ratio of the minimum
    distance to a centroid of a different class to the distance to its own class
    centroid.
    """
    x_penultimate = evaluations["x_penultimate"]
    y_label = evaluations["y_label"]
    unique_classes = np.unique(y_label)
    class_centroids = {
        label: np.mean(x_penultimate[y_label == label], axis=0)
        for label in unique_classes
    }

    distance_ratios = []
    for sample, label in zip(x_penultimate, y_label):
        true_centroid = class_centroids[label]
        distance_to_centroid = np.linalg.norm(sample - true_centroid)
        distances_to_others = [
            np.linalg.norm(sample - centroid)
            for other, centroid in class_centroids.items() if other != label
        ]
        min_distance_to_other = min(distances_to_others) if distances_to_others else 0.0
        ratio = min_distance_to_other / distance_to_centroid if distance_to_centroid > 0 else 0.0
        distance_ratios.append(ratio)

    distance_ratios_array = np.array(distance_ratios)
    return {
        "distance_ratios": distance_ratios_array,
        "average_distance_ratio": np.mean(distance_ratios_array),
    }


def get_binarity_metrics(evaluations):
    """
    Calculate binarity metrics using Gaussian Mixture Models for each feature dimension.

    For each dimension in the penultimate layer, a two-component GMM is fit and various
    metrics (GMM score, peaks distance, etc.) are computed.
    """
    x_penultimate = evaluations["x_penultimate"]
    scores = []
    stds_list = []
    peaks_distance_list = []
    n_features = x_penultimate.shape[1]

    for d in range(n_features):
        feature = x_penultimate[:, d].reshape(-1, 1)
        scaled_feature = StandardScaler().fit_transform(feature)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(n_components=2)
                gmm.fit(scaled_feature)
        except Exception as e:
            raise RuntimeError("Error in GMM fit for feature {}: {}".format(d, e)) from e

        scores.append(gmm.score(scaled_feature))
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        stds_list.append(stds)
        avg_std = np.mean(stds)
        peaks_distance_list.append(np.abs(means[0] - means[1]) / avg_std if avg_std != 0 else 0.0)

    scores = np.array(scores)
    stds_list = np.array(stds_list)
    peaks_distance_list = np.array(peaks_distance_list)

    score_mean = np.mean(scores)
    stds_mean = np.mean(stds_list)
    peaks_distance_mean = np.mean(peaks_distance_list)
    score_min = np.min(scores)
    peaks_distance_min = np.min(peaks_distance_list)
    arg_score_av = int(np.argmin(np.abs(scores - score_mean)))
    arg_peaks_distance_av = int(np.argmin(np.abs(peaks_distance_list - peaks_distance_mean)))

    return {
        "score_mean": score_mean,
        "score_min": score_min,
        "stds_mean": stds_mean,
        "peaks_distance_mean": peaks_distance_mean,
        "peaks_distance_min": peaks_distance_min,
        "arg_score_av": arg_score_av,
        "arg_peaks_distance_av": arg_peaks_distance_av,
    }
