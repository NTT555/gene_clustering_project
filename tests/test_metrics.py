import unittest
import numpy as np
from src.metrics import calculate_euclidean_distance, calculate_pearson_distance

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Dữ liệu test giả lập
        # Khoảng cách Euclidean từ (0,0) đến (3,4) chắc chắn phải bằng 5 (Pytago)
        self.X_euclidean = np.array([
            [0.0, 0.0],
            [3.0, 4.0],
            [0.0, 5.0]
        ])
        
        # Hai vector tương quan hoàn hảo (r=1) và tương quan ngược (r=-1)
        self.X_pearson = np.array([
            [1, 2, 3],       
            [2, 4, 6],       # Tương quan hoàn toàn với dòng 0 (r=1)
            [3, 2, 1]        # Tương quan ngược với dòng 0 (r=-1)
        ])

    def test_euclidean_distance(self):
        dist_matrix = calculate_euclidean_distance(self.X_euclidean)
        
        # 1. Kiểm tra hình dáng ma trận (N x N)
        self.assertEqual(dist_matrix.shape, (3, 3))
        
        # 2. Khoảng cách với chính nó phải bằng 0
        np.testing.assert_almost_equal(np.diag(dist_matrix), np.zeros(3))
        
        # 3. Tính chính xác toán học (d(0,1) = 5)
        self.assertAlmostEqual(dist_matrix[0, 1], 5.0)
        self.assertAlmostEqual(dist_matrix[0, 2], 5.0)
        
        # 4. Tính đối xứng: d(A,B) phải bằng d(B,A)
        self.assertAlmostEqual(dist_matrix[0, 1], dist_matrix[1, 0])

    def test_pearson_distance(self):
        dist_matrix = calculate_pearson_distance(self.X_pearson)
        
        # Khoảng cách Pearson: d = 1 - r
        # Dòng 0 và Dòng 1 giống hệt nhau về xu hướng (r=1) -> d = 0
        self.assertAlmostEqual(dist_matrix[0, 1], 0.0)
        
        # Dòng 0 và Dòng 2 ngược hướng hoàn toàn (r=-1) -> d = 1 - (-1) = 2
        self.assertAlmostEqual(dist_matrix[0, 2], 2.0)

if __name__ == '__main__':
    unittest.main()