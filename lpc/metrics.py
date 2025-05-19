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



def _compute_gradient_norm_batch(model: nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, device: torch.device):
    """
    Helper function to compute the L2 norm of the gradient of the logit difference
    (true_logit - max_other_logit) w.r.t. the input for a single batch using torch.autograd.grad.

    Args:
        model: The neural network model.
        x_batch: Input batch tensor.
        y_batch: True label batch tensor.
        device: Device to run computations on.

    Returns:
        Numpy array of gradient norms for each sample in the batch.
    """
    model.eval() # Ensure model is in evaluation mode
    # Ensure requires_grad is set for the input batch
    x_batch = x_batch.clone().detach().to(device).requires_grad_(True)
    y_batch = y_batch.to(device)

    # Ensure model parameters do not require gradients during this specific calculation
    # if we only care about gradient w.r.t. input x_batch
    original_param_req_grad = {}
    for name, param in model.named_parameters():
        original_param_req_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Forward pass to get logits
    # Ensure requires_grad is enabled for the forward pass w.r.t. x_batch
    with torch.enable_grad():
        outputs = model(x_batch)

    # Restore original requires_grad state for model parameters
    for name, param in model.named_parameters():
        param.requires_grad_(original_param_req_grad[name])


    # Handle potential dictionary output from model wrappers
    if isinstance(outputs, dict) and 'x_output' in outputs:
        logits = outputs['x_output']
    elif isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        # Attempt to restore grad state before raising error
        for name, param in model.named_parameters():
             param.requires_grad_(original_param_req_grad[name])
        raise TypeError(f"Unexpected model output type: {type(outputs)}")

    if logits.shape[0] != y_batch.shape[0]:
        warnings.warn(f"Logits batch size ({logits.shape[0]}) != Labels batch size ({y_batch.shape[0]}). Skipping batch.")
        # Restore original requires_grad state for model parameters before returning
        for name, param in model.named_parameters():
             param.requires_grad_(original_param_req_grad[name])
        return np.array([], dtype=np.float32)

    batch_size = x_batch.shape[0]
    num_classes = logits.shape[1]

    # --- Efficient Batch Calculation ---
    # Get indices for true class logits
    true_label_indices = y_batch
    true_logits = logits[torch.arange(batch_size), true_label_indices]

    # Create a mask to find max_other_logit
    # Set true class logits to a very small number temporarily
    masked_logits = logits.clone()
    masked_logits[torch.arange(batch_size), true_label_indices] = -float('inf')

    # Get the highest logit among other classes for each sample
    max_other_logits, _ = masked_logits.max(dim=1)

    # Compute the difference (margin relative to highest competitor) for all samples
    logit_diffs = true_logits - max_other_logits

    # Check for cases where there was only one class (max_other_logits is -inf)
    # In such cases, the gradient is technically zero.
    valid_diffs_mask = torch.isfinite(logit_diffs)
    if not valid_diffs_mask.all():
        warnings.warn(f"Found non-finite logit differences (possibly due to single-class samples). Setting gradient norm to 0 for these.")
        # We'll handle setting norms to 0 later based on the mask

    # Compute gradients of the logit differences w.r.t the input batch x_batch
    # We compute the gradient of the sum, which is equivalent to the sum of gradients.
    # Use grad_outputs=torch.ones_like(logit_diffs) if computing grad of vector output.
    # Computing grad of sum is often simpler. Only compute for valid diffs.
    if valid_diffs_mask.any(): # Proceed only if there's at least one valid diff
        # Zero out previous gradients on x_batch if any lingered
        if x_batch.grad is not None:
            x_batch.grad.zero_()

        # Calculate gradients only for the valid differences sum
        grads = torch.autograd.grad(
            outputs=logit_diffs[valid_diffs_mask].sum(), # Sum of valid logit differences
            inputs=x_batch,
            # grad_outputs=torch.ones_like(logit_diffs[valid_diffs_mask]), # Use if outputs=logit_diffs[valid_diffs_mask]
            retain_graph=False,  # No need to retain graph after this grad calculation
            create_graph=False,  # We are not doing higher-order derivatives
            allow_unused=True    # Allow if some part of x_batch didn't affect output (shouldn't happen here)
        )[0] # Get the gradients w.r.t. x_batch (the first input specified)
    else:
        grads = None # No valid gradients to compute

    grad_norms = np.zeros(batch_size, dtype=np.float32)

    if grads is not None:
        # Calculate L2 norm for each sample's gradient
        # Grads tensor shape: (batch_size, C, H, W) or (batch_size, features)
        # Reshape to (batch_size, -1) to compute norm per sample
        grads_flat_per_sample = grads.view(batch_size, -1)
        norms_tensor = torch.linalg.norm(grads_flat_per_sample, dim=1)

        # Assign computed norms, respecting the valid_diffs_mask
        # Ensure norms_tensor corresponds only to the valid inputs if grads were computed subsetted (they weren't here, grads shape matches x_batch)
        grad_norms[valid_diffs_mask.cpu().numpy()] = norms_tensor.cpu().numpy()
        # Norms for invalid diffs remain 0.0 as initialized

    else:
        # All diffs were invalid, all norms remain 0.0
        warnings.warn("All logit differences were non-finite. All gradient norms set to 0.")


    # Detach input after use
    x_batch.requires_grad_(False)

    return grad_norms



def estimate_lipschitz_gradient_norm_stream(model: nn.Module, loader: torch.utils.data.DataLoader, max_batches: int = None, device: torch.device = None):
    """
    Estimates the Lipschitz constant by finding the maximum observed L2 norm
    of the gradient of the logit difference (true_logit - max_other_logit)
    with respect to the input, evaluated iteratively over batches from a DataLoader.

    Memory-friendly version for large datasets.

    Args:
        model: The neural network model.
        loader: DataLoader yielding batches of (inputs, labels).
        max_batches: Maximum number of batches to process. Processes all if None.
        device: The torch device ('cuda', 'cpu', etc.) to use. Auto-detects if None.

    Returns:
        Dictionary containing:
            - 'max_gradient_norm': The maximum observed gradient norm across processed batches.
            - 'average_gradient_norm': The average observed gradient norm across processed samples.
            - 'num_samples_used': Total number of samples processed.
            - 'num_batches_used': Total number of batches processed.
    """
    # Corrected indentation
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval() # Ensure model is in evaluation mode

    all_grad_norms = []
    num_samples_processed = 0
    num_batches_processed = 0

    print(f"Estimating gradient norms using DataLoader...")
    # Removed tqdm wrapper
    with torch.no_grad(): # Temporarily disable grad globally for efficiency, enable locally
        batch_iterator = loader # Use the loader directly
        for batch_idx, batch_data in enumerate(batch_iterator):
            # Corrected indentation
            if max_batches is not None and batch_idx >= max_batches:
                print(f"Reached max_batches limit ({max_batches}). Stopping.")
                break

            # --- Input Data Handling ---
            # Try to handle different DataLoader return types
            # Corrected indentation
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                x_batch, y_batch = batch_data[0], batch_data[1]
                # Add further checks if necessary (e.g., tensor type)
            elif isinstance(batch_data, dict) and 'image' in batch_data and 'label' in batch_data:
                 x_batch, y_batch = batch_data['image'], batch_data['label']
            else:
                 warnings.warn(f"Skipping batch {batch_idx}: Unexpected data format from DataLoader: {type(batch_data)}")
                 continue # Corrected indentation

            # Corrected indentation
            if not isinstance(x_batch, torch.Tensor) or not isinstance(y_batch, torch.Tensor):
                 warnings.warn(f"Skipping batch {batch_idx}: Expected tensors, got {type(x_batch)}, {type(y_batch)}")
                 continue # Corrected indentation
            # --- End Input Data Handling ---


            # Ensure labels are Long type
            y_batch = y_batch.long()

            # Enable gradient computation locally for this batch
            with torch.enable_grad():
                batch_grad_norms = _compute_gradient_norm_batch(model, x_batch, y_batch, device)

            if len(batch_grad_norms) > 0:
                all_grad_norms.extend(batch_grad_norms)
                num_samples_processed += len(batch_grad_norms)

            num_batches_processed += 1



    # Corrected indentation
    if not all_grad_norms:
        warnings.warn("No valid gradient norms were computed across all batches. Returning zero estimates.")
        max_norm = 0.0
        avg_norm = 0.0
    else:
        all_grad_norms_np = np.array(all_grad_norms)
        # Filter out potential NaNs or zeros if warnings occurred
        valid_norms = all_grad_norms_np[np.isfinite(all_grad_norms_np) & (all_grad_norms_np > 1e-9)]

        # Corrected indentation
        if len(valid_norms) == 0:
            warnings.warn("All computed gradient norms were non-finite or zero. Returning zero estimates.")
            max_norm = 0.0
            avg_norm = 0.0
        else:
            max_norm = np.max(valid_norms)
            avg_norm = np.mean(valid_norms)


    return {
        "max_gradient_norm": float(max_norm),
        "average_gradient_norm": float(avg_norm),
        "num_samples_used": num_samples_processed,
        "num_batches_used": num_batches_processed,
    }

