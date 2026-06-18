"""
src/visualization.py
=====================
Thanh vien 4: Data Visualization

Nhiem vu:
    - plot_dendrogram(): ve cay phan cum (dendrogram) tu linkage matrix
    - plot_biclustering_heatmap(): ve heatmap ket hop voi dendrogram hai chieu
      (cluster ca genes va ca samples)

Input cua module nay chinh la output cua Thanh vien 3 (src/clustering.py):
    - linkage_matrix: numpy array dang chuan scipy linkage, shape (n_samples-1, 4)
      moi dong la [idx1, idx2, distance, sample_count]
    - data: ma tran numpy (genes x samples) da duoc chuan hoa, dung de ve heatmap

Tat ca hinh anh output se duoc luu vao thu muc results/.
"""

import os

import matplotlib
matplotlib.use("Agg")  # khong can man hinh, chi luu file -> tranh loi khi chay tren server/CI
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Cau hinh chung
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _ensure_results_dir(results_dir: str = RESULTS_DIR) -> str:
    """Tao thu muc results/ neu chua ton tai."""
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ---------------------------------------------------------------------------
# 1. plot_dendrogram
# ---------------------------------------------------------------------------
def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels=None,
    title: str = "Hierarchical Clustering Dendrogram",
    color_threshold: float = None,
    orientation: str = "top",
    figsize: tuple = (12, 6),
    save_path: str = None,
    show: bool = False,
):
    """
    Ve dendrogram tu linkage matrix (output cua Thanh vien 3).

    Tham so
    -------
    linkage_matrix : np.ndarray
        Ma tran lien ket dang chuan scipy, shape (n-1, 4).
        Day chinh la output tu ham AgglomerativeClustering() cua Thanh vien 3.
    labels : list[str], optional
        Ten cac mau/gen tuong ung voi tung diem du lieu goc.
        Neu None, scipy se tu danh so thu tu.
    title : str
        Tieu de bieu do.
    color_threshold : float, optional
        Nguong khoang cach de to mau cac nhanh khac nhau (phan biet cluster).
        Neu None, scipy tu chon nguong mac dinh.
    orientation : str
        Huong ve: "top", "bottom", "left", "right".
    figsize : tuple
        Kich thuoc hinh (rong, cao) tinh bang inch.
    save_path : str, optional
        Duong dan luu file. Neu None, tu dong luu vao results/dendrogram.png
    show : bool
        True neu muon hien thi hinh ngay (plt.show()), thuong dat False khi chay batch.

    Tra ve
    ------
    dict : ket qua tra ve tu scipy.cluster.hierarchy.dendrogram
           (chua thong tin thu tu la, mau sac... co the dung lai cho heatmap)
    """
    if linkage_matrix is None or len(linkage_matrix) == 0:
        raise ValueError(
            "linkage_matrix dang rong. Hay chac chan da nhan duoc output "
            "tu ham AgglomerativeClustering() cua Thanh vien 3."
        )

    fig, ax = plt.subplots(figsize=figsize)

    ddata = dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=color_threshold,
        orientation=orientation,
        ax=ax,
        leaf_rotation=90 if orientation in ("top", "bottom") else 0,
        leaf_font_size=8,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    if orientation in ("top", "bottom"):
        ax.set_xlabel("Samples / Genes")
        ax.set_ylabel("Distance")
    else:
        ax.set_xlabel("Distance")
        ax.set_ylabel("Samples / Genes")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path is None:
        results_dir = _ensure_results_dir()
        save_path = os.path.join(results_dir, "dendrogram.png")
    else:
        _ensure_results_dir(os.path.dirname(save_path) or RESULTS_DIR)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[visualization] Da luu dendrogram tai: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return ddata


# ---------------------------------------------------------------------------
# 2. plot_biclustering_heatmap
# ---------------------------------------------------------------------------
def plot_biclustering_heatmap(
    data: np.ndarray,
    row_linkage: np.ndarray = None,
    col_linkage: np.ndarray = None,
    row_labels=None,
    col_labels=None,
    title: str = "Biclustering Heatmap",
    cmap: str = "RdBu_r",
    figsize: tuple = (12, 12),
    save_path: str = None,
    show: bool = False,
):
    """
    Ve heatmap ket hop voi dendrogram (biclustering: cluster ca hang va cot).

    Day la bieu do kinh dien trong phan tich gene expression: hang la genes,
    cot la samples (hoac nguoc lai), mau sac the hien muc do bieu hien gen,
    kem 2 dendrogram o canh tren va canh trai de minh hoa cau truc cum.

    Tham so
    -------
    data : np.ndarray
        Ma tran du lieu da chuan hoa (vd: Z-score), shape (n_genes, n_samples).
        Day la du lieu da xu ly tu Thanh vien 1 (data_preprocessing.py).
    row_linkage : np.ndarray, optional
        Linkage matrix cho hang (thuong la genes). Neu None, khong ve dendrogram hang.
    col_linkage : np.ndarray, optional
        Linkage matrix cho cot (thuong la samples). Neu None, khong ve dendrogram cot.
    row_labels, col_labels : list[str], optional
        Nhan cho hang/cot.
    title : str
        Tieu de bieu do.
    cmap : str
        Bang mau (mac dinh "RdBu_r": xanh = bieu hien thap, do = bieu hien cao).
    figsize : tuple
        Kich thuoc hinh.
    save_path : str, optional
        Duong dan luu file. Neu None, tu dong luu vao results/biclustering_heatmap.png
    show : bool
        True neu muon hien thi hinh ngay.

    Tra ve
    ------
    matplotlib.figure.Figure
    """
    if data is None or data.size == 0:
        raise ValueError("Du lieu dau vao (data) dang rong.")

    n_rows, n_cols = data.shape

    # Bo cuc luoi: dendrogram tren (cot), dendrogram trai (hang), heatmap chinh, colorbar
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 4, 0.2],
        height_ratios=[1, 4],
        wspace=0.02, hspace=0.02,
    )

    ordered_data = data
    row_order = np.arange(n_rows)
    col_order = np.arange(n_cols)

    # --- Dendrogram cot (tren) ---
    ax_col_dendro = fig.add_subplot(gs[0, 1])
    if col_linkage is not None and len(col_linkage) > 0:
        col_ddata = dendrogram(col_linkage, ax=ax_col_dendro, no_labels=True,
                                color_threshold=0, above_threshold_color="#444444")
        col_order = np.array(col_ddata["leaves"])
    ax_col_dendro.axis("off")

    # --- Dendrogram hang (trai) ---
    ax_row_dendro = fig.add_subplot(gs[1, 0])
    if row_linkage is not None and len(row_linkage) > 0:
        row_ddata = dendrogram(row_linkage, ax=ax_row_dendro, orientation="left",
                                no_labels=True, color_threshold=0,
                                above_threshold_color="#444444")
        row_order = np.array(row_ddata["leaves"])
    ax_row_dendro.axis("off")

    # --- Sap xep lai du lieu theo thu tu cluster ---
    ordered_data = data[np.ix_(row_order, col_order)]

    # --- Heatmap chinh ---
    ax_heatmap = fig.add_subplot(gs[1, 1])
    im = ax_heatmap.imshow(ordered_data, aspect="auto", cmap=cmap,
                            interpolation="nearest")

    if col_labels is not None:
        ax_heatmap.set_xticks(np.arange(n_cols))
        ax_heatmap.set_xticklabels(
            [col_labels[i] for i in col_order], rotation=90, fontsize=7
        )
    else:
        ax_heatmap.set_xticks([])

    if row_labels is not None and n_rows <= 60:
        # chi hien nhan hang khi so gen khong qua nhieu, neu khong se roi
        ax_heatmap.set_yticks(np.arange(n_rows))
        ax_heatmap.set_yticklabels(
            [row_labels[i] for i in row_order], fontsize=5.5
        )
        ax_heatmap.tick_params(axis="y", pad=2)
    else:
        ax_heatmap.set_yticks([])

    ax_heatmap.set_xlabel("Samples")
    ax_heatmap.set_ylabel("Genes")

    # --- Colorbar ---
    ax_cbar = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Expression level (Z-score)", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if save_path is None:
        results_dir = _ensure_results_dir()
        save_path = os.path.join(results_dir, "biclustering_heatmap.png")
    else:
        _ensure_results_dir(os.path.dirname(save_path) or RESULTS_DIR)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[visualization] Da luu biclustering heatmap tai: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Demo / self-test khi chay truc tiep file nay
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Phan nay CHI dung de Thanh vien 4 tu kiem tra code khi cac module cua
    Thanh vien 2 (metrics.py) va Thanh vien 3 (clustering.py) chua san sang.

    Mock: tu tinh linkage bang scipy de gia lap dung output ma Thanh vien 3
    se tra ve (cung format), nham dam bao 2 ham tren chay dung truoc khi
    ghep vao pipeline thuc cua main.py.
    """
    import pandas as pd
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    print("=== DEMO / SELF-TEST cho visualization.py (Thanh vien 4) ===")

    # 1. Doc du lieu gen mau va chuan hoa nhe de demo
    csv_path = "/mnt/user-data/uploads/data_set_ALL_AML_independent.csv"
    df = pd.read_csv(csv_path)

    # chi lay cac cot gia tri bieu hien gen (bo cot "call")
    value_cols = [c for c in df.columns if c not in ("Gene Description", "Gene Accession Number")
                  and not c.startswith("call")]

    # lay mau 40 gen dau (de hinh ve khong qua nang) va chuan hoa Z-score theo hang
    sample_df = df.iloc[:40][value_cols].astype(float)
    gene_names = df.iloc[:40]["Gene Accession Number"].tolist()
    sample_names = value_cols

    matrix = sample_df.to_numpy()
    z = (matrix - matrix.mean(axis=1, keepdims=True)) / (matrix.std(axis=1, keepdims=True) + 1e-9)

    # 2. Mock output cua Thanh vien 3: tu tinh linkage cho hang (genes) va cot (samples)
    row_link = linkage(pdist(z, metric="euclidean"), method="average")
    col_link = linkage(pdist(z.T, metric="euclidean"), method="average")

    # 3. Goi 2 ham chinh cua Thanh vien 4
    plot_dendrogram(
        row_link,
        labels=gene_names,
        title="Demo Dendrogram - Gene Clustering (mock data)",
        save_path=os.path.join(_ensure_results_dir(), "demo_dendrogram.png"),
    )

    plot_biclustering_heatmap(
        z,
        row_linkage=row_link,
        col_linkage=col_link,
        row_labels=gene_names,
        col_labels=sample_names,
        title="Demo Biclustering Heatmap (mock data)",
        save_path=os.path.join(_ensure_results_dir(), "demo_biclustering_heatmap.png"),
    )

    print("=== Hoan tat. Kiem tra anh trong thu muc results/ ===")
