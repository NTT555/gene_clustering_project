# 🧬 From-Scratch Hierarchical Clustering for Gene Expression Analysis

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains a production-ready, modular implementation of **Agglomerative Hierarchical Clustering from scratch** (without using `scikit-learn` for clustering core logic). The engine is applied to a high-dimensional microarray dataset (`data_set_ALL_AML_independent.csv`) to classify leukemia patients into distinct biological sub-groups (**ALL** - Acute Lymphoblastic Leukemia vs. **AML** - Acute Myeloid Leukemia) and discover co-expressed gene signatures.

## 🏗️ Project Architecture
The project is strictly modularized adhering to professional software engineering practices:

```text
gene_clustering_project/
│
├── data/
│   └── raw/
│       └── data_set_ALL_AML_independent.csv  # Raw gene expression matrix
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                # Ingestion, Z-score, and Variance filtering
│   ├── metrics.py                           # Vectorized Euclidean & Pearson distances
│   ├── clustering.py                        # From-scratch Agglomerative Clustering engine
│   ├── visualization.py                     # Plotting logic for Dendrograms & Heatmaps
│   └── evaluation.py                        # Automated K selection (Elbow & Silhouette)
│
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py                      # Unit tests for distance metrics
│   └── test_clustering.py                   # Unit tests for clustering engine
│
├── results/                                 # Generated visual artifacts (.png)
├── .gitignore                               # Cache and local environment exclusion
├── Main.py                                  # Production pipeline entry point
└── Hierarchical_Clustering_Analysis.ipynb   # Comprehensive research notebook
```

## 🧠 Mathematical Foundations

### 1. Vectorized Distance Metrics
To optimize computational performance and bypass inefficient Python loops, pairwise distances are computed via heavy matrix algebra utilizing **NumPy Vectorization**:
* **Euclidean Distance (Patient Space):** Computes geometric distances by expanding the binomial square matrix-wise: 
  $$||A - B||^2 = A^2 + B^2 - 2AB$$
* **Pearson Correlation Distance (Gene Space):** Standardizes profiles to identify functional co-expression patterns independent of absolute scales:
  $$d = 1 - r$$

### 2. Agglomerative Engine & Lance-Williams Formulations
The clustering engine builds the entire dendrogram hierarchy from bottom-up. After merging two clusters $C_i$ and $C_j$ into a new cluster $C_k$, the distances to any other cluster $C_m$ are dynamically updated via the **Lance-Williams formula** without recomputing original pairwise points:

* **Average Linkage (Default):**
  $$d(C_k, C_m) = \frac{|C_i| \cdot d(C_i, C_m) + |C_j| \cdot d(C_j, C_m)}{|C_i| + |C_j|}$$
* **Ward's Linkage:** Minimizes variance accumulation within clusters.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/NTT555/gene_clustering_project.git
cd gene_clustering_project
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### Running the Production Pipeline
To execute the entire end-to-end processing, clustering, evaluation, and visualization pipeline, run:
```bash
python main.py
```

### Running Automated Unit Tests
To verify the mathematical integrity of the core engines, run the test suites:
```bash
python -m unittest discover -s tests
```

## 📊 Model Evaluation & Stopping Criteria
Determining the optimal number of clusters ($K$) is performed automatically via an explicit dual-axis evaluation script (`src/evaluation.py`):
1.  **Elbow Method:** Tracks the monotonic increase in merge distances across the linkage matrix $Z$. A sharp structural "knee/elbow" indicates the saturation point of natural boundaries.
2.  **Silhouette Analysis:** Computes intra-cluster cohesion versus inter-cluster separation to benchmark clustering quality.

The pipeline successfully validates $K=2$ as the mathematical optimal choice, aligning seamlessly with the biological reality of the **ALL/AML** patient split.

## 🎨 Visual Artifacts
Upon execution, the following production-grade plots are exported to the `results/` directory:
* `patients_dendro.png`: Hierarchical tree structure classifying patient cohorts.
* `genes_dendro.png`: Hierarchy highlighting genetic correlation networks.
* `patients_evaluation_k.png`: Combined Line Chart showing Elbow and Silhouette results.
* `biclustering_heatmap.png`: A comprehensive dual-clustered heatmap revealing clean checkboard blocks (Gene Signatures) representing genetic biomarkers for Leukemia classification.
