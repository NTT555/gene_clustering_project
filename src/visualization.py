import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import logging

logger = logging.getLogger(__name__)

def plot_dendrogram(linkage_matrix, labels, title, save_path):
    plt.figure(figsize=(15, 8))
    plt.title(title, fontsize=16)
    plt.xlabel('Chỉ mục / Tên mẫu', fontsize=12)
    plt.ylabel('Khoảng cách', fontsize=12)
    
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90., leaf_font_size=10.)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Đã lưu: {save_path}")

def plot_heatmap(data, row_linkage, col_linkage, row_labels, col_labels, title, save_path):
    cg = sns.clustermap(
        data,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        yticklabels=row_labels,
        xticklabels=col_labels,
        cmap='coolwarm',
        figsize=(12, 10)
    )
    cg.fig.suptitle(title, fontsize=16, y=1.05)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Đã lưu: {save_path}")