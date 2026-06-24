import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw gene expression data and remove non-numeric metadata columns.
    
    Args:
        filepath (str): Path to the raw CSV data file.
        
    Returns:
        pd.DataFrame: Cleaned numerical matrix with genes as rows and samples as columns.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Drop assay quality indicator columns (e.g., 'call' containing A, P, M)
    cols_to_drop = [col for col in df.columns if 'call' in col.lower()]
    df_clean = df.drop(columns=cols_to_drop)
    
    # Set proper identifiers as index
    if 'Gene Accession Number' in df_clean.columns:
        df_clean = df_clean.set_index('Gene Accession Number')
        
    if 'Gene Description' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Gene Description'])
        
    # Coerce to numeric to handle any accidental string artifacts
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    logger.info(f"Data cleaning completed. Matrix shape: {df_clean.shape}")
    return df_clean

def normalize_zscore(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """
    Apply Z-score normalization along the specified axis.
    
    Args:
        df (pd.DataFrame): Raw expression matrix.
        axis (int): 1 for row-wise (gene) normalization, 0 for column-wise (sample).
        
    Returns:
        pd.DataFrame: Z-score normalized expression matrix.
    """
    logger.info(f"Applying Z-score normalization along axis {axis}")
    
    if axis == 1:
        mu = df.mean(axis=1)
        sigma = df.std(axis=1, ddof=1).replace(0, 1e-8)
        return df.sub(mu, axis=0).div(sigma, axis=0)
    else:
        mu = df.mean(axis=0)
        sigma = df.std(axis=0, ddof=1).replace(0, 1e-8)
        return df.sub(mu, axis=1).div(sigma, axis=1)

def filter_top_genes_by_variance(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Select the top N most highly variable genes for dimensionality reduction.
    
    Args:
        df (pd.DataFrame): Normalized expression matrix.
        top_n (int): Number of top features to retain.
        
    Returns:
        pd.DataFrame: Subset of the expression matrix.
    """
    variances = df.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(top_n).index
    logger.info(f"Retained top {top_n} genes based on variance thresholding.")
    return df.loc[top_genes]