import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

def plot_clustering_evaluation(X: np.ndarray, Z: np.ndarray, max_k: int, export_path: str) -> None:
    """
    Evaluate optimal number of clusters (K) using Elbow Method and Silhouette Score.
    
    Args:
        X (np.ndarray): Original feature matrix.
        Z (np.ndarray): Linkage matrix from hierarchical clustering.
        max_k (int): Maximum number of clusters to evaluate (e.g., 10).
        export_path (str): Path to save the evaluation plot.
    """
    logger.info(f"Evaluating optimal K (from 2 to {max_k})...")
    
    k_values = range(2, max_k + 1)
    merge_distances = []
    silhouette_scores = []

    for k in k_values:
        # 1. Elbow Method: Extract merge distance that reduces cluster count from K to K-1
        # Z[-1] is the final merge (2 to 1 cluster). Z[-(k-1)] is the merge from k to k-1.
        dist = Z[-(k - 1), 2]
        merge_distances.append(dist)

        # 2. Silhouette Score: Cut the dendrogram to get exactly 'k' clusters
        labels = fcluster(Z, k, criterion='maxclust')
        
        # Calculate silhouette score for the current partition
        score = silhouette_score(X, labels, metric='euclidean')
        silhouette_scores.append(score)

    # --- Plotting the Dual-Axis Chart ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: Elbow Method (Merge Distance)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Merge Distance (Elbow)', color=color1, fontsize=12)
    ax1.plot(k_values, merge_distances, marker='o', color=color1, linewidth=2, markersize=8, label='Elbow (Distance)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Axis 2: Silhouette Score
    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color2, fontsize=12)
    ax2.plot(k_values, silhouette_scores, marker='s', color=color2, linewidth=2, markersize=8, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Aesthetics and titles
    plt.title('Optimal K Evaluation: Elbow Method vs Silhouette Score', fontsize=14, pad=15)
    fig.tight_layout()
    
    # Save the artifact
    plt.savefig(export_path, dpi=300)
    plt.close()
    logger.info(f"Evaluation chart exported successfully to: {export_path}")