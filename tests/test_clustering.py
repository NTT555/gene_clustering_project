import unittest
import numpy as np
from src.clustering import AgglomerativeClustering
from src.metrics import calculate_euclidean_distance

class TestHierarchicalClustering(unittest.TestCase):
    """Unit tests for the custom Agglomerative Clustering engine."""
    
    def setUp(self):
        """Initialize a 1D dataset with explicit proximity clusters."""
        # P0 and P1 are closest (distance 1), followed by P2 and P3 (distance 2)
        self.X = np.array([[1], [2], [10], [12]])

    def test_linkage_matrix_structural_integrity(self):
        """Ensure the output Z matrix complies with SciPy structural standards."""
        model = AgglomerativeClustering(linkage='single')
        Z = model.fit(self.X, calculate_euclidean_distance)
        
        n_samples = self.X.shape[0]
        self.assertEqual(Z.shape, (n_samples - 1, 4))
        
        # Initial merge must involve exactly 2 elements
        self.assertEqual(Z[0, 3], 2)
        # Terminal merge must encompass all elements
        self.assertEqual(Z[-1, 3], n_samples)

    def test_monotonicity_and_merge_logic(self):
        """Verify that merge distances increase monotonically and logic holds."""
        model = AgglomerativeClustering(linkage='single')
        Z = model.fit(self.X, calculate_euclidean_distance)
        
        distances = Z[:, 2]
        # Check monotonic increase in linkage distance
        is_monotonic = all(x <= y for x, y in zip(distances, distances[1:]))
        self.assertTrue(is_monotonic, "Linkage distances are not monotonically increasing.")

if __name__ == '__main__':
    unittest.main()