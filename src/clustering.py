import numpy as np
import logging

logger = logging.getLogger(__name__)

class AgglomerativeClustering:
    def __init__(self, linkage='average'):
        self.linkage = linkage.lower()
        if self.linkage not in ['single', 'complete', 'average', 'ward']:
            raise ValueError("Chỉ hỗ trợ: 'single', 'complete', 'average', 'ward'.")

    def fit(self, X, metric_function):
        n_samples = X.shape[0]
        logger.info(f"Bắt đầu gom cụm {n_samples} mẫu bằng phương pháp {self.linkage}...")
        
        # 1. Khởi tạo ma trận khoảng cách
        dist_matrix = metric_function(X)
        np.fill_diagonal(dist_matrix, np.inf)

        # Ma trận Z lưu lịch sử gộp cụm: [id_1, id_2, distance, sample_count]
        Z = np.zeros((n_samples - 1, 4))
        
        cluster_sizes = {i: 1 for i in range(n_samples)}
        matrix_idx_to_cluster_id = {i: i for i in range(n_samples)}
        current_cluster_id = n_samples

        for step in range(n_samples - 1):
            if step % 50 == 0 and step > 0:
                logger.info(f"Đã gộp {step}/{n_samples-1} bước...")

            # 2. Tìm cặp cụm gần nhất
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            idx_1, idx_2 = min_idx
            if idx_1 > idx_2:
                idx_1, idx_2 = idx_2, idx_1

            id_1 = matrix_idx_to_cluster_id[idx_1]
            id_2 = matrix_idx_to_cluster_id[idx_2]
            min_dist = dist_matrix[idx_1, idx_2]

            # 3. Ghi vào ma trận Linkage Z
            size_new = cluster_sizes[id_1] + cluster_sizes[id_2]
            Z[step] = [id_1, id_2, min_dist, size_new]
            cluster_sizes[current_cluster_id] = size_new

            # 4. Cập nhật khoảng cách bằng công thức Lance-Williams
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

            # Cập nhật lại vào ma trận
            dist_matrix[idx_1, :] = new_distances
            dist_matrix[:, idx_1] = new_distances
            dist_matrix[idx_1, idx_1] = np.inf
            matrix_idx_to_cluster_id[idx_1] = current_cluster_id

            # Vô hiệu hóa cụm đã gộp
            dist_matrix[idx_2, :] = np.inf
            dist_matrix[:, idx_2] = np.inf
            
            current_cluster_id += 1

        return Z