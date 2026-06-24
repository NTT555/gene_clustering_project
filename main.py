import os
import logging

# Core algorithms and data processing modules
from src.data_preprocessing import load_and_clean_data, normalize_zscore, filter_top_genes_by_variance
from src.metrics import calculate_euclidean_distance, calculate_pearson_distance
from src.clustering import AgglomerativeClustering

# Visualization module
from src.visualization import generate_dendrogram, generate_biclustering_heatmap

# Configure standard logging for the pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

from src.evaluation import plot_clustering_evaluation

# Define project directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data_set_ALL_AML_independent.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    logging.info("--- INITIATING HIERARCHICAL CLUSTERING PIPELINE ---")
    
    # 1 & 2. Data Ingestion and Z-score Normalization
    df_raw = load_and_clean_data(DATA_PATH)
    df_norm = normalize_zscore(df_raw, axis=1)
    
    # 3. Feature Selection (Top 50 Highly Variable Genes)
    df_subset = filter_top_genes_by_variance(df_norm, top_n=50)
    
    # 4. Sample/Patient Clustering (Columns)
    logging.info("Executing Patient/Sample Clustering...")
    X_patients = df_subset.values.T 
    patient_labels = df_subset.columns.tolist()
    
    hc_patients = AgglomerativeClustering(linkage='average')
    Z_patients = hc_patients.fit(X_patients, calculate_euclidean_distance)
    
    # Delegate dendrogram rendering to the visualization module
    generate_dendrogram(
        Z_patients, 
        patient_labels, 
        "Patient Dendrogram (Euclidean + Average)", 
        os.path.join(RESULTS_DIR, "patients_dendro.png")
    )

    plot_clustering_evaluation(
        X=X_patients, 
        Z=Z_patients, 
        max_k=10, 
        export_path=os.path.join(RESULTS_DIR, "patients_evaluation_k.png")
    )

    # 5. Feature/Gene Clustering (Rows)
    logging.info("Executing Gene Feature Clustering...")
    X_genes = df_subset.values
    gene_labels = df_subset.index.tolist()
    
    hc_genes = AgglomerativeClustering(linkage='average')
    Z_genes = hc_genes.fit(X_genes, calculate_pearson_distance)
    
    # Delegate dendrogram rendering to the visualization module
    generate_dendrogram(
        Z_genes, 
        gene_labels, 
        "Gene Dendrogram (Pearson + Average)", 
        os.path.join(RESULTS_DIR, "genes_dendro.png")
    )

    # 6. Render Comprehensive Biclustering Heatmap
    logging.info("Rendering comprehensive Biclustering Heatmap...")
    generate_biclustering_heatmap(
        df_subset, 
        Z_genes, 
        Z_patients, 
        gene_labels, 
        patient_labels, 
        "Gene Expression Biclustering Heatmap", 
        os.path.join(RESULTS_DIR, "biclustering_heatmap.png")
    )

    logging.info(f"Pipeline Execution Terminated. Artifacts exported to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()