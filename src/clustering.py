import numpy as np
import logging

logger = logging.getLogger(__name__)

class AgglomerativeClustering:
    """
    Custom Implementation of Agglomerative Hierarchical Clustering.
    Constructs a linkage matrix iteratively using the generalized Lance-Williams update formula.
    """
    
    def __init__(self, linkage: str = 'average'):
        """
        Initialize the clustering configuration.
        
        Args:
            linkage (str): Criterion for merging ('single', 'complete', 'average', 'ward').
        """
        self.linkage = linkage.lower()
        valid_linkages = ['single', 'complete', 'average', 'ward']
        if self.linkage not in valid_linkages:
            raise ValueError(f"Supported linkages are: {valid_linkages}")

    def fit(self, X: np.ndarray, metric_func: callable) -> np.ndarray:
        """
        Fit the hierarchical clustering model.
        
        Args:
            X (np.ndarray): Feature matrix.
            metric_func (callable): Distance function reference.
            
        Returns:
            np.ndarray: Linkage matrix Z of shape (n_samples - 1, 4).
        """
        n_samples = X.shape[0]
        logger.info(f"Initializing {self.linkage.capitalize()} Linkage Clustering for {n_samples} samples.")
        
        # Initialize primary distance matrix
        dist_matrix = metric_func(X)
        np.fill_diagonal(dist_matrix, np.inf)

        # Z matrix structure required by SciPy: [cluster1, cluster2, distance, new_cluster_size]
        Z = np.zeros((n_samples - 1, 4))
        
        # State tracking dictionaries
        cluster_sizes = {i: 1 for i in range(n_samples)}
        matrix_idx_to_cluster_id = {i: i for i in range(n_samples)}
        next_cluster_id = n_samples

        # Agglomerative O(N^3) procedural loop
        for step in range(n_samples - 1):
            # Locate the global minimum distance pair
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            idx_1, idx_2 = min_idx
            
            # Enforce idx_1 < idx_2 strictly for structural consistency
            if idx_1 > idx_2:
                idx_1, idx_2 = idx_2, idx_1

            id_1 = matrix_idx_to_cluster_id[idx_1]
            id_2 = matrix_idx_to_cluster_id[idx_2]
            min_dist = dist_matrix[idx_1, idx_2]

            # Register the merge event in the linkage matrix Z
            new_size = cluster_sizes[id_1] + cluster_sizes[id_2]
            Z[step] = [id_1, id_2, min_dist, new_size]
            cluster_sizes[next_cluster_id] = new_size

            # Dynamic update of distance matrix via Lance-Williams coefficients
            new_distances = np.zeros(dist_matrix.shape[0])
            for i in range(dist_matrix.shape[0]):
                if i != idx_1 and i != idx_2:
                    d_si = dist_matrix[i, idx_1]
                    d_sj = dist_matrix[i, idx_2]
                    
                    s_i = cluster_sizes[id_1]
                    s_j = cluster_sizes[id_2]
                    s_k = cluster_sizes[matrix_idx_to_cluster_id[i]]
                    
                    if self.linkage == 'single':
                        d_new = min(d_si, d_sj)
                    elif self.linkage == 'complete':
                        d_new = max(d_si, d_sj)
                    elif self.linkage == 'average':
                        d_new = (s_i * d_si + s_j * d_sj) / (s_i + s_j)
                    elif self.linkage == 'ward':
                        total = s_i + s_j + s_k
                        d_new = np.sqrt(((s_i + s_k) * d_si**2 + (s_j + s_k) * d_sj**2 - s_k * min_dist**2) / total)
                    
                    new_distances[i] = d_new

            # Integrate the newly formed cluster into the distance matrix structure
            dist_matrix[idx_1, :] = new_distances
            dist_matrix[:, idx_1] = new_distances
            dist_matrix[idx_1, idx_1] = np.inf
            matrix_idx_to_cluster_id[idx_1] = next_cluster_id

            # Deactivate the subsumed cluster vector
            dist_matrix[idx_2, :] = np.inf
            dist_matrix[:, idx_2] = np.inf
            
            next_cluster_id += 1

        return Z