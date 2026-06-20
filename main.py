import os
import logging
from src.data_preprocessing import load_and_clean_data, normalize_zscore, filter_top_genes_by_variance
from src.metrics import calculate_euclidean_distance, calculate_pearson_distance
from src.clustering import AgglomerativeClustering
from src.visualization import plot_dendrogram, plot_heatmap

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "data_set_ALL_AML_independent.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("="*60)
    print("HIERARCHICAL CLUSTERING FROM SCRATCH - KHÔNG DÙNG SKLEARN")
    print("="*60)

    print("\n[1] ĐỌC VÀ LÀM SẠCH DỮ LIỆU...")
    df_raw = load_and_clean_data(DATA_PATH)
    
    print("\n[2] CHUẨN HÓA & LỌC GEN...")
    df_norm = normalize_zscore(df_raw, axis=1)
    df_subset = filter_top_genes_by_variance(df_norm, top_n=50)
    
    print("\n[3] GOM CỤM BỆNH NHÂN (BẰNG EUCLIDEAN)...")
    X_patients = df_subset.values.T 
    patient_labels = df_subset.columns.tolist()
    
    model_patient = AgglomerativeClustering(linkage='average')
    linkage_patients = model_patient.fit(X_patients, calculate_euclidean_distance)
    
    plot_dendrogram(
        linkage_patients, patient_labels,
        "Dendrogram - Phân nhóm Bệnh nhân",
        os.path.join(RESULTS_DIR, "dendrogram_patients.png")
    )

    print("\n[4] GOM CỤM GEN (BẰNG PEARSON)...")
    X_genes = df_subset.values
    gene_labels = df_subset.index.tolist()
    
    model_gene = AgglomerativeClustering(linkage='average')
    linkage_genes = model_gene.fit(X_genes, calculate_pearson_distance)

    plot_dendrogram(
        linkage_genes, gene_labels,
        "Dendrogram - Phân nhóm Gen",
        os.path.join(RESULTS_DIR, "dendrogram_genes.png")
    )

    print("\n[5] VẼ HEATMAP TỔNG HỢP...")
    plot_heatmap(
        df_subset, linkage_genes, linkage_patients, gene_labels, patient_labels,
        "Biclustering Heatmap: Top 50 Genes vs Patients",
        os.path.join(RESULTS_DIR, "clustermap_genes_patients.png")
    )

    print("\n" + "="*60)
    print(f" HOÀN TẤT! KẾT QUẢ ĐÃ LƯU TẠI: {RESULTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()