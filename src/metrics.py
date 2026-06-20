import numpy as np

def calculate_euclidean_distance(data: np.ndarray) -> np.ndarray:
    """Tính ma trận khoảng cách Euclidean (O(N^2) vectorized)."""
    # Khai triển: (A - B)^2 = A^2 + B^2 - 2AB
    A_sq = np.sum(data**2, axis=1, keepdims=True)
    B_sq = np.sum(data**2, axis=1, keepdims=True).T
    AB = np.dot(data, data.T)
    
    distances_sq = np.clip(A_sq + B_sq - 2 * AB, a_min=0.0, a_max=None)
    return np.sqrt(distances_sq)

def calculate_pearson_distance(data: np.ndarray) -> np.ndarray:
    """Tính ma trận khoảng cách Pearson: d = 1 - r"""
    correlation_matrix = np.corrcoef(data)
    dist_matrix = 1.0 - correlation_matrix
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix