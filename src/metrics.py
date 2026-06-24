import numpy as np

def calculate_euclidean_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance using optimized vectorized operations.
    Formula utilized: (A - B)^2 = A^2 + B^2 - 2AB
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Symmetric distance matrix of shape (n_samples, n_samples).
    """
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    # Broadcasting to construct the N x N distance matrix
    sq_distances = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
    
    # Clip to mitigate floating-point inaccuracies yielding infinitesimal negative values
    sq_distances = np.clip(sq_distances, a_min=0.0, a_max=None)
    
    return np.sqrt(sq_distances)

def calculate_pearson_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Pearson correlation distance (d = 1 - r).
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Symmetric distance matrix of shape (n_samples, n_samples).
    """
    correlation_matrix = np.corrcoef(X)
    distance_matrix = 1.0 - correlation_matrix
    
    # Ensure zero distance along the principal diagonal
    np.fill_diagonal(distance_matrix, 0.0)
    
    return distance_matrix