import unittest
import numpy as np
from src.metrics import calculate_euclidean_distance, calculate_pearson_distance

class TestDistanceMetrics(unittest.TestCase):
    """Unit tests for vectorized distance matrix computations."""
    
    def setUp(self):
        """Initialize mock matrices for validation."""
        self.X_euclidean = np.array([
            [0.0, 0.0],
            [3.0, 4.0],
            [0.0, 5.0]
        ])
        self.X_pearson = np.array([
            [1, 2, 3],       
            [2, 4, 6],       # Perfectly positively correlated with row 0
            [3, 2, 1]        # Perfectly negatively correlated with row 0
        ])

    def test_euclidean_computation(self):
        """Validate Euclidean distance outputs against known geometric configurations."""
        dist_matrix = calculate_euclidean_distance(self.X_euclidean)
        
        self.assertEqual(dist_matrix.shape, (3, 3))
        # Distance from origin (0,0) to (3,4) must be exactly 5.0
        self.assertAlmostEqual(dist_matrix[0, 1], 5.0)
        # Verify symmetry
        self.assertAlmostEqual(dist_matrix[0, 1], dist_matrix[1, 0])
        # Diagonal must be zero
        np.testing.assert_almost_equal(np.diag(dist_matrix), np.zeros(3))

    def test_pearson_computation(self):
        """Validate Pearson correlation distance logic (d = 1 - r)."""
        dist_matrix = calculate_pearson_distance(self.X_pearson)
        
        # Identical correlation (r=1) implies distance 0
        self.assertAlmostEqual(dist_matrix[0, 1], 0.0)
        # Inverse correlation (r=-1) implies maximum distance 2
        self.assertAlmostEqual(dist_matrix[0, 2], 2.0)

if __name__ == '__main__':
    unittest.main()