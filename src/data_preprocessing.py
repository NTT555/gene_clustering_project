import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Đọc dữ liệu gen gốc và loại bỏ các cột không chứa giá trị số."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    logger.info("Đang đọc dữ liệu từ: %s", filepath)
    df = pd.read_csv(filepath)
    
    # Loại bỏ các cột 'call' (chứa A, P, M)
    cols_to_drop = [c for c in df.columns if 'call' in c.lower()]
    df_clean = df.drop(columns=cols_to_drop)
    
    if 'Gene Accession Number' in df_clean.columns:
        df_clean = df_clean.set_index('Gene Accession Number')
    if 'Gene Description' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Gene Description'])
        
    logger.info("Kích thước ma trận biểu hiện: %s hàng (gen) × %s cột (bệnh nhân)", *df_clean.shape)
    
    # Ép kiểu về số thực
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_clean

def normalize_zscore(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """Chuẩn hóa Z-score hoàn toàn bằng Pandas/Numpy (Không dùng sklearn)."""
    logger.info(f"Chuẩn hóa Z-score theo {'gen (hàng)' if axis==1 else 'bệnh nhân (cột)'}...")
    
    if axis == 1:
        mu = df.mean(axis=1)
        sigma = df.std(axis=1, ddof=1).replace(0, 1e-8)
        df_norm = df.sub(mu, axis=0).div(sigma, axis=0)
    else:
        mu = df.mean(axis=0)
        sigma = df.std(axis=0, ddof=1).replace(0, 1e-8)
        df_norm = df.sub(mu, axis=1).div(sigma, axis=1)
        
    return df_norm

def filter_top_genes_by_variance(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Lọc ra Top N gen có phương sai lớn nhất để giảm chiều dữ liệu."""
    variances = df.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(top_n).index
    logger.info(f"Đã lọc Top {top_n} gen có phương sai cao nhất.")
    return df.loc[top_genes]