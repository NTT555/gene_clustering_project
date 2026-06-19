import os

import matplotlib
matplotlib.use("Agg")  # không cần màn hình, chỉ lưu file -> tránh lỗi khi chạy trên server/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


# ===========================================================================
# CẤU HÌNH THƯ MỤC
# ===========================================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _ensure_results_dir(results_dir: str = RESULTS_DIR) -> str:
    """Tạo thư mục results/ nếu chưa tồn tại. An toàn với path rỗng."""
    if not results_dir:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ===========================================================================
# 1. plot_dendrogram
# ===========================================================================
def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels=None,
    title: str = "Dendrogram",
    color_threshold: float = None,
    figsize: tuple = (12, 6),
    save_path: str = None,
):
    """
    Vẽ biểu đồ cây phân cụm (Dendrogram) từ linkage matrix.

    Tham số
    -------
    linkage_matrix : np.ndarray, shape (n-1, 4)
        Output trực tiếp từ AgglomerativeClustering().fit() của Thành viên 3.
    labels : list[str], optional
        Tên mẫu/gen tương ứng với từng điểm dữ liệu gốc (vd: tên bệnh nhân
        hoặc Gene Accession Number). Nếu None, scipy tự đánh số.
    title : str
        Tiêu đề biểu đồ.
    color_threshold : float, optional
        Ngưỡng khoảng cách để tô màu phân biệt các nhánh/cụm.
        Nếu None, toàn bộ cây vẽ 1 màu (an toàn, không phụ thuộc vào scipy
        tự đoán ngưỡng).
    figsize : tuple
        Kích thước hình (rộng, cao) tính bằng inch.
    save_path : str, optional
        Đường dẫn lưu file. Nếu None, tự động lưu vào results/dendrogram.png

    Trả về
    ------
    dict : kết quả trả về từ scipy.cluster.hierarchy.dendrogram
    """
    if linkage_matrix is None or len(linkage_matrix) == 0:
        raise ValueError(
            "linkage_matrix đang rỗng. Hãy đảm bảo đã nhận đúng output "
            "từ AgglomerativeClustering().fit() của Thành viên 3."
        )

    plt.figure(figsize=figsize)

    ddata = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Mẫu / Gen", fontsize=12)
    plt.ylabel("Khoảng cách", fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_results_dir(), "dendrogram.png")
    else:
        _ensure_results_dir(os.path.dirname(save_path))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[visualization] Đã lưu Dendrogram tại: {save_path}")
    plt.close()

    return ddata


# ===========================================================================
# 2. plot_biclustering_heatmap
# ===========================================================================
def plot_biclustering_heatmap(
    data,
    row_linkage: np.ndarray = None,
    col_linkage: np.ndarray = None,
    row_labels="auto",
    col_labels="auto",
    title: str = "Biclustering Heatmap",
    cmap: str = "coolwarm",
    figsize: tuple = (10, 10),
    top_n_genes: int = None,
    save_path: str = None,
):
    """
    Vẽ Heatmap kết hợp dendrogram 2 chiều (genes và samples) bằng seaborn.clustermap.

    Tham số
    -------
    data : pd.DataFrame hoặc np.ndarray, shape (n_genes, n_samples)
        Ma trận biểu hiện gen đã chuẩn hoá (vd: expr_norm từ data_preprocessing.py
        của Thành viên 1). Hàng = genes, cột = samples.
    row_linkage : np.ndarray, optional
        Linkage matrix cho hàng (genes), từ AgglomerativeClustering của TV3.
        Nếu None, seaborn sẽ KHÔNG tự tính lại (tránh nhầm thuật toán) —
        truyền vào row_cluster=False.
    col_linkage : np.ndarray, optional
        Linkage matrix cho cột (samples), từ AgglomerativeClustering của TV3.
        Nếu None, tương tự không cluster cột.
    row_labels, col_labels : list[str] hoặc "auto"
        Nhãn cho hàng/cột. "auto" sẽ lấy tên từ index/columns nếu data là DataFrame.
    title : str
        Tiêu đề biểu đồ.
    cmap : str
        Bảng màu (mặc định "coolwarm": xanh = biểu hiện thấp, đỏ = biểu hiện cao).
    figsize : tuple
        Kích thước hình.
    top_n_genes : int, optional
        Nếu được set, chỉ vẽ N gen có phương sai (variance) cao nhất giữa các
        mẫu — giúp heatmap dễ đọc và nhanh hơn khi số gen quá lớn (>200).
        Khuyến nghị dùng khi vẽ trên toàn bộ tập gen đã lọc (thường 1000+ gen).
        Lưu ý: nếu dùng tham số này CÙNG với row_linkage có sẵn, row_linkage
        phải được tính lại tương ứng với đúng tập con N gen này (xem phần
        demo ở cuối file để biết cách làm).
    save_path : str, optional
        Đường dẫn lưu file. Nếu None, tự động lưu vào results/biclustering_heatmap.png

    Trả về
    ------
    seaborn.matrix.ClusterGrid
    """
    if data is None or len(data) == 0:
        raise ValueError("Dữ liệu đầu vào (data) đang rỗng.")

    # Chuẩn hoá data về DataFrame để dễ thao tác nhãn
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Lọc top N gen có variance cao nhất (nếu được yêu cầu)
    if top_n_genes is not None and top_n_genes < data.shape[0]:
        variances = data.var(axis=1)
        top_idx = variances.sort_values(ascending=False).head(top_n_genes).index
        data = data.loc[top_idx]
        print(
            f"[visualization] top_n_genes={top_n_genes}: đã lọc còn "
            f"{data.shape[0]} gen có phương sai cao nhất để vẽ heatmap."
        )
        if row_linkage is not None and len(row_linkage) != data.shape[0] - 1:
            print(
                "[visualization] CẢNH BÁO: row_linkage được truyền vào không khớp "
                "số gen sau khi lọc top_n_genes. Bỏ qua row_linkage, sẽ không "
                "cluster theo hàng. Hãy tính lại linkage trên đúng tập con gen này."
            )
            row_linkage = None

    row_cluster = row_linkage is not None
    col_cluster = col_linkage is not None

    g = sns.clustermap(
        data,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        figsize=figsize,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.8, 0.05, 0.18),
    )

    # Xoay nhãn cho dễ đọc
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)

    g.fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    if save_path is None:
        save_path = os.path.join(_ensure_results_dir(), "biclustering_heatmap.png")
    else:
        _ensure_results_dir(os.path.dirname(save_path))

    g.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[visualization] Đã lưu Heatmap tại: {save_path}")
    plt.close(g.fig)

    return g


# ===========================================================================
# DEMO / SELF-TEST — chạy thử với pipeline thật của cả nhóm
# ===========================================================================
if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_preprocessing import load_data, handle_missing_values, normalize_data
    from metrics import calculate_euclidean_distance
    from clustering import AgglomerativeClustering

    print("=== DEMO: chạy thử visualization.py với pipeline thật ===")

    # 1. Đọc + tiền xử lý dữ liệu (TV1)
    #    LƯU Ý: filter_genes() của TV1 hiện có lỗi lệch index (boolean mask
    #    không khớp index khi lọc DataFrame). Nhóm cần TV1 sửa lại hàm này
    #    (ví dụ reset_index trước khi tạo mask, hoặc set lại index thống nhất).
    #    Demo dưới đây tạm thời bỏ qua filter_genes() để không bị chặn,
    #    và lọc top gen theo variance ngay trong bước vẽ (top_n_genes).
    df = load_data()
    df_clean = handle_missing_values(df, strategy="median")
    expr_norm, _ = normalize_data(df_clean, axis=0)
    print("Dữ liệu sau chuẩn hoá:", expr_norm.shape, "(genes x samples)")

    # 2. Chọn top gen biến thiên nhiều nhất để demo nhanh (thay cho filter_genes)
    TOP_N = 50
    variances = expr_norm.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(TOP_N).index
    expr_subset = expr_norm.loc[top_genes]
    print(f"Chọn top {TOP_N} gen có phương sai cao nhất để demo.")

    # 3. Tính linkage cho GENES (hàng) và SAMPLES (cột) bằng code thật của TV2 + TV3
    t0 = time.time()
    model = AgglomerativeClustering(linkage="average")

    row_link = model.fit(expr_subset.values, calculate_euclidean_distance)
    print(f"Linkage genes: {row_link.shape}, thời gian: {time.time()-t0:.2f}s")

    t0 = time.time()
    col_link = model.fit(expr_subset.values.T, calculate_euclidean_distance)
    print(f"Linkage samples: {col_link.shape}, thời gian: {time.time()-t0:.2f}s")

    # 4. Vẽ dendrogram cho samples (34 bệnh nhân)
    plot_dendrogram(
        col_link,
        labels=expr_subset.columns.tolist(),
        title="Dendrogram - Phân cụm bệnh nhân (ALL/AML)",
        save_path=os.path.join(_ensure_results_dir(), "demo_dendrogram_samples.png"),
    )

    # 5. Vẽ biclustering heatmap (top 50 gen x 34 samples)
    plot_biclustering_heatmap(
        expr_subset,
        row_linkage=row_link,
        col_linkage=col_link,
        title=f"Biclustering Heatmap - Top {TOP_N} gen biến thiên cao nhất",
        save_path=os.path.join(_ensure_results_dir(), "demo_biclustering_heatmap.png"),
    )

    print("=== Hoàn tất. Kiểm tra ảnh trong thư mục results/ ===")
