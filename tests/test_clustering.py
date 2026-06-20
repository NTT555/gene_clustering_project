import unittest
import numpy as np
from src.clustering import AgglomerativeClustering
from src.metrics import calculate_euclidean_distance

class TestClustering(unittest.TestCase):
    def setUp(self):
        # Tập dữ liệu 1D cực kỳ đơn giản:
        # P0=1, P1=2, P2=10, P3=12
        # Rõ ràng P0 và P1 gần nhất (cách 1), P2 và P3 gần nhì (cách 2)
        self.X = np.array([[1], [2], [10], [12]])

    def test_linkage_matrix_structure(self):
        model = AgglomerativeClustering(linkage='single')
        Z = model.fit(self.X, calculate_euclidean_distance)
        
        N = self.X.shape[0]
        
        # 1. Kiểm tra hình dáng ma trận Z phải là (N-1, 4)
        self.assertEqual(Z.shape, (N - 1, 4))
        
        # 2. Cột thứ 4 (index 3) lưu số lượng phần tử của cụm mới. 
        # Bước gộp đầu tiên luôn là 2 phần tử đơn gộp lại -> size = 2
        self.assertEqual(Z[0, 3], 2)
        
        # Bước gộp cuối cùng phải chứa toàn bộ N phần tử
        self.assertEqual(Z[-1, 3], N)

    def test_clustering_logic(self):
        model = AgglomerativeClustering(linkage='single')
        Z = model.fit(self.X, calculate_euclidean_distance)
        
        # Bước 1: P0(1) và P1(2) phải được gộp trước vì khoảng cách d=1 là nhỏ nhất
        # ID của cụm tạo ra sẽ là 4 (do N=4, ID gốc là 0,1,2,3)
        self.assertTrue(Z[0, 0] == 0 or Z[0, 1] == 0)
        self.assertTrue(Z[0, 0] == 1 or Z[0, 1] == 1)
        self.assertAlmostEqual(Z[0, 2], 1.0) # Khoảng cách gộp = 1
        
        # Cột khoảng cách (index 2) bắt buộc phải tăng dần một cách đơn điệu (Monotonic)
        # Vì các cụm càng về sau thì khoảng cách càng xa nhau
        distances = Z[:, 2]
        is_monotonic_increasing = all(x <= y for x, y in zip(distances, distances[1:]))
        self.assertTrue(is_monotonic_increasing, "Khoảng cách trong ma trận Z không tăng dần!")

if __name__ == '__main__':
    unittest.main()