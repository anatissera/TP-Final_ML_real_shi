from typing import Dict, List
import numpy as np
tqdm = __import__('tqdm').tqdm
import copy


def trim_to_full_batches(arrays: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """
    Trims each array in `arrays` so that its first dimension is divisible by batch_size.
    All arrays must have the same number of samples.

    Returns a list of trimmed arrays.
    """
    n_samples = arrays[0].shape[0]
    n_trimmed = (n_samples // batch_size) * batch_size
    trimmed = []
    for arr in arrays:
        assert arr.shape[0] == n_samples, "All inputs must have same sample count"
        trimmed.append(arr[:n_trimmed])
    return trimmed


def ensure_3d(x: np.ndarray) -> np.ndarray:
    """
    Ensures that `x` has shape (samples, time, leads).
    If it's 2D (samples, time), adds a singleton last dimension.
    """
    if x.ndim == 2:
        return x[..., np.newaxis]
    return x


def count_per_class(labels: np.ndarray, num_classes: int) -> List[int]:
    """Returns counts of each class index in [0, num_classes)."""
    return [int(np.sum(labels == i)) for i in range(num_classes)]


def preprocess_data_fmm(input_data: Dict[str, np.ndarray],
                        dataset_params: Dict[str, List[int]],
                        fs: int,
                        batch_size: int,
                        split_ecg: bool = False,
                        **kwargs) -> Dict[str, np.ndarray]:
    """
    Preprocesses FMM data for model ingestion:
    - Casts data to float32
    - Deep copies data and labels to avoid side-effects
    - Ensures 3D shape
    - Trims to full batches
    - Reports class distribution

    Returns a dict with keys:
    'data', 'labels', 'sizes', 'coefficients', 'coefficients_ang'
    """

    # Extract and copy inputs
    data = copy.deepcopy(input_data['data'].astype(np.float32))
    labels = copy.deepcopy(input_data['labels'])
    sizes = input_data['sizes']
    coeffs = input_data['coefficients']
    coeffs_ang = input_data['coefficients_ang']

    # Ensure proper dimensions
    data = ensure_3d(data)

    # Trim to full batches
    data, labels, sizes, coeffs, coeffs_ang = trim_to_full_batches(
        [data, labels, sizes, coeffs, coeffs_ang], batch_size
    )

    # Compute class distribution
    classes = dataset_params.get('classes', []) if isinstance(dataset_params, dict) else []
    counts = count_per_class(labels, num_classes=len(classes))
    print(f"Number of samples per class: {counts}")

    return {
        'data': data,
        'labels': labels,
        'sizes': sizes,
        'coefficients': coeffs,
        'coefficients_ang': coeffs_ang
    }