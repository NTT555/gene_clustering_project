# 🧬 Gene Clustering Project (ALL vs AML)

## 📌 Giới thiệu

Dự án này thực hiện **phân cụm gene expression** nhằm phân tích và khám phá sự khác biệt giữa hai loại bệnh:

* ALL (Acute Lymphoblastic Leukemia)
* AML (Acute Myeloid Leukemia)

Pipeline bao gồm:

* Tiền xử lý dữ liệu
* Chuẩn hóa (Z-score)
* Lọc gene theo phương sai
* Tính khoảng cách
* Phân cụm phân cấp (Hierarchical Clustering)
* Trực quan hóa (Dendrogram, Heatmap)

---

## 📂 Cấu trúc thư mục

```
gene_clustering_project/
├── data/
│   ├── raw/                      # Chứa dữ liệu gen gốc (không bao giờ sửa file ở đây)
│   └── processed/                # Chứa dữ liệu đã qua làm sạch và chuẩn hóa
├── src/                          # Thư mục chứa mã nguồn cốt lõi
│   ├── __init__.py
│   ├── data_preprocessing.py     # Code đọc, lọc gen, chuẩn hóa Z-score
│   ├── metrics.py                # Code tính ma trận khoảng cách (Euclidean, Pearson)
│   ├── clustering.py             # Code thuật toán Hierarchical, Lance-Williams
│   └── visualization.py          # Code vẽ Dendrogram và Heatmap
├── notebooks/
│   └── biological_analysis.ipynb # Jupyter Notebook để phân tích kết quả sinh học
├── tests/                        # Thư mục chứa các kịch bản kiểm thử
│   ├── __init__.py
│   ├── test_metrics.py           # Test tính chính xác của hàm khoảng cách
│   └── test_clustering.py        # Test logic gộp cụm xem có đúng thứ tự không
├── results/                      # Nơi lưu các hình ảnh biểu đồ, file CSV kết quả
├── main.py                       # File trung tâm để chạy toàn bộ quy trình từ A-Z
├── requirements.txt              # Danh sách thư viện (numpy, pandas, matplotlib, seaborn...)
├── .gitignore                    # Các file bỏ qua không đưa lên hệ thống 
└── README.md                     # Tài liệu hướng dẫn cài đặt và chạy project

```

---

## ⚙️ Cài đặt

### 1. Clone repo

```bash
git clone <your-repo-url>
cd gene_clustering_project
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

---

## ▶️ Cách chạy

### ✔️ Chạy bằng script

```bash
python main.py
```

### ✔️ Chạy bằng Jupyter Notebook

```bash
jupyter notebook
```

Sau đó mở:

```
notebooks/gene_clustering_notebook.ipynb
```

---

## 📊 Pipeline xử lý

1. **Load dữ liệu**
2. **Clean dữ liệu**

   * Loại bỏ cột không phải số
3. **Chuẩn hóa Z-score**
4. **Chọn top gene theo variance**
5. **Tính khoảng cách**

   * Euclidean
   * Pearson
6. **Phân cụm (Agglomerative Clustering)**
7. **Visualization**

   * Dendrogram
   * Heatmap

---

## 📈 Ví dụ sử dụng

```python
from data_preprocessing import load_and_clean_data, normalize_zscore
from metrics import calculate_euclidean_distance
from clustering import AgglomerativeClustering

df = load_and_clean_data("data/raw/data_set_ALL_AML_independent.csv")
df_norm = normalize_zscore(df)

distance_matrix = calculate_euclidean_distance(df_norm)

model = AgglomerativeClustering()
model.fit(distance_matrix)
```

---

## ⚠️ Lỗi thường gặp

### ❌ ModuleNotFoundError

👉 Fix:

```python
import sys
sys.path.append("src")
```

---

### ❌ FileNotFoundError

👉 Kiểm tra path:

```python
import os
print(os.getcwd())
```

---

## 🧠 Ý nghĩa sinh học

* Giúp phân biệt ALL vs AML dựa trên gene expression
* Tìm ra các gene quan trọng có độ biến thiên cao
* Hỗ trợ nghiên cứu ung thư và chẩn đoán

---

## 🚀 Hướng phát triển

* Thêm PCA / t-SNE visualization
* Áp dụng K-means / DBSCAN
* Feature selection nâng cao
* Tích hợp machine learning classifier
