import numpy as np

class AgglomerativeClustering:
    def __init__(self, linkage='single'):
        """
        Khởi tạo bộ phân cụm phân cấp.
        
        Parameters:
        -----------
        linkage : str, tùy chọn (mặc định='single')
            Phương pháp tính khoảng cách giữa các cụm.
            Hỗ trợ: 'single' (MIN), 'complete' (MAX), 'average' (UPGMA).
        """
        self.linkage = linkage.lower()
        if self.linkage not in ['single', 'complete', 'average']:
            raise ValueError(f"Phương pháp linkage '{linkage}' không được hỗ trợ. Chọn 'single', 'complete', hoặc 'average'.")

    def _get_lance_williams_params(self, method):
        """
        Trả về các hệ số alpha_i, alpha_j, beta, gamma cho công thức Lance-Williams:
        d(u, v) = alpha_i * d(s, i) + alpha_j * d(s, j) + beta * d(i, j) + gamma * |d(s, i) - d(s, j)|
        """
        # Các hệ số này tạm thời để cấu hình tĩnh, việc tính toán động dựa vào kích thước cụm
        # sẽ được xử lý trực tiếp trong hàm cập nhật.
        return method

    def fit(self, X, metric_function):
        """
        Thực hiện phân cụm và xây dựng Linkage Matrix.
        
        Parameters:
        -----------
        X : np.ndarray
            Ma trận dữ liệu gen đầu vào kích thước (n_samples, n_features).
        metric_function : function
            Hàm tính ma trận khoảng cách từ `src/metrics.py` của Thành viên 2.
            Ví dụ: calculate_euclidean_distance(X) -> trả về ma trận (n_samples, n_samples)
            
        Returns:
        --------
        linkage_matrix : np.ndarray
            Ma trận liên kết kích thước (n_samples - 1, 4).
            Mỗi dòng chứa: [cụm_1, cụm_2, khoảng_cách, số_lượng_phần_tử_trong_cụm_mới]
        """
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Dữ liệu phải có ít nhất 2 mẫu để phân cụm.")

        # Tạo một bản sao để tránh chỉnh sửa dữ liệu gốc
        dist_matrix = metric_function(X).astype(float)
        
        # Đặt đường chéo chính bằng vô cùng để tránh tự gộp chính nó
        np.fill_diagonal(dist_matrix, np.inf)

        # Ban đầu mỗi sample là một cụm riêng biệt từ 0 đến n_samples - 1
        # cluster_sizes lưu số lượng phần tử của từng cụm (kể cả cụm mới tạo thành)
        cluster_sizes = {i: 1 for i in range(n_samples)}
        
        # Danh sách các cụm hiện tại đang hoạt động
        active_clusters = list(range(n_samples))
        
        # Định danh cho cụm mới được tạo ra (bắt đầu từ n_samples)
        next_cluster_id = n_samples
        
        # Ma trận kết quả Linkage Matrix đúng chuẩn Scipy
        linkage_matrix = []

        for step in range(n_samples - 1):
            min_dist = np.inf
            cluster_i_idx = -1
            cluster_j_idx = -1

            # Tìm cặp cụm có khoảng cách nhỏ nhất trong số các cụm đang hoạt động
            n_active = len(active_clusters)
            for s1 in range(n_active):
                for s2 in range(s1 + 1, n_active):
                    c1 = active_clusters[s1]
                    c2 = active_clusters[s2]
                    
                    if dist_matrix[c1, c2] < min_dist:
                        min_dist = dist_matrix[c1, c2]
                        cluster_i_idx = s1
                        cluster_j_idx = s2

            # Lấy ID thực tế của 2 cụm cần gộp (đảm bảo i < j để thống nhất quy ước)
            c_i = active_clusters[cluster_i_idx]
            c_j = active_clusters[cluster_j_idx]
            if c_i > c_j:
                c_i, c_j = c_j, c_i

            # Tính số lượng phần tử của cụm mới
            new_size = cluster_sizes[c_i] + cluster_sizes[c_j]
            cluster_sizes[next_cluster_id] = new_size

            # Ghi lại thông tin gộp vào Linkage Matrix
            linkage_matrix.append([float(c_i), float(c_j), float(min_dist), float(new_size)])

            # Mở rộng ma trận khoảng cách để chuẩn bị cho cụm mới (thêm 1 dòng, 1 cột)
            # Kích thước cũ là next_cluster_id x next_cluster_id
            new_dist_matrix = np.inf * np.ones((next_cluster_id + 1, next_cluster_id + 1))
            new_dist_matrix[:next_cluster_id, :next_cluster_id] = dist_matrix
            dist_matrix = new_dist_matrix

            # Loại bỏ c_i và c_j khỏi danh sách hoạt động, thêm cụm mới vào
            active_clusters.remove(c_i)
            active_clusters.remove(c_j)

            for s in active_clusters:
                d_si = dist_matrix[s, c_i]
                d_sj = dist_matrix[s, c_j]
                d_ij = min_dist # Khoảng cách giữa 2 cụm vừa gộp

                if self.linkage == 'single':
                    # Single Linkage (MIN): d(s, new) = 0.5*d(s,i) + 0.5*d(s,j) - 0.5*|d(s,i) - d(s,j)|
                    d_new = 0.5 * d_si + 0.5 * d_sj - 0.5 * abs(d_si - d_sj)
                    
                elif self.linkage == 'complete':
                    # Complete Linkage (MAX): d(s, new) = 0.5*d(s,i) + 0.5*d(s,j) + 0.5*|d(s,i) - d(s,j)|
                    d_new = 0.5 * d_si + 0.5 * d_sj + 0.5 * abs(d_si - d_sj)
                    
                elif self.linkage == 'average':
                    # Average Linkage (UPGMA): d(s, new) = (|i|*d(s,i) + |j|*d(s,j)) / (|i| + |j|)
                    n_i = cluster_sizes[c_i]
                    n_j = cluster_sizes[c_j]
                    d_new = (n_i * d_si + n_j * d_sj) / (n_i + n_j)

                dist_matrix[s, next_cluster_id] = d_new
                dist_matrix[next_cluster_id, s] = d_new

            # Kích hoạt cụm mới
            active_clusters.append(next_cluster_id)
            next_cluster_id += 1

        return np.array(linkage_matrix)
