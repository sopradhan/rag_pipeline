"""
nw.py
-----
Refactored clustering pipeline demo.

This script is structured as a reusable pipeline that accepts raw data (2D or multi-dimensional)
and runs the following steps:

1. Preprocess (scaling)
2. Run PCA and decide a sensible clustering "k" from a set of candidate values using
   PCA explained variance (or number of components needed for a target variance)
3. Run clustering (KMeans by default, but DBSCAN or others may be used)
4. Produce new features: cluster labels and PCA components appended to the feature set
5. Visualize results (for 2D datasets) or save outputs

How k is adjusted using PCA (the strategy implemented):
- Compute PCA on the scaled data and determine the number of components required to reach
  a chosen explained-variance threshold (e.g. 90%). Call this `pca_n_components`.
- From a candidate list of k values (e.g. [6,5,4,10]), pick the k that is numerically closest
  to `pca_n_components`. This is a simple heuristic that links intrinsic dimensionality
  to a reasonable number of clusters. You can change the heuristic to any other rule.

To use with your own data: replace the `load_sample_data()` call with your dataset (NumPy
array of shape [n_samples, n_features]) and call `run_pipeline(X, candidate_ks=[...])`.

Author: Automated refactor
"""

from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

try:
    from sentence_transformers import SentenceTransformer
    SENTE = True
except Exception:
    SENTE = False


def load_sample_data(n_samples: int = 500, noise: float = 0.05, random_state: int = 42):
    """Return a simple 2D circular dataset for demo and testing.

    Replace this function with a loader that returns your raw features (X) and
    optional labels if available.
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    return X, y


# -----------------
# Flexible data loaders
# -----------------

def load_from_csv(path: str, feature_columns: Optional[List[str]] = None) -> np.ndarray:
    """Load numeric features from a CSV file. If feature_columns is None,
    numeric columns will be used automatically.
    Returns numpy array X.
    """
    df = pd.read_csv(path)
    if feature_columns:
        X = df[feature_columns].values
    else:
        X = df.select_dtypes(include=[np.number]).values
    return X


def load_from_sqlite(db_path: str, query: str = 'SELECT * FROM data', feature_columns: Optional[List[str]] = None) -> np.ndarray:
    """Run a SQL query on a SQLite DB and return feature matrix.
    For other DBs, replace sqlite3 with appropriate driver (psycopg2, pyodbc, etc.).
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    if feature_columns:
        X = df[feature_columns].values
    else:
        X = df.select_dtypes(include=[np.number]).values
    return X


def load_from_json_payload(payload: List[dict], feature_columns: Optional[List[str]] = None) -> np.ndarray:
    """Convert a list of JSON objects (dicts) into a feature matrix.
    Non-numeric columns will be dropped unless feature_columns specified.
    """
    df = pd.DataFrame(payload)
    if feature_columns:
        X = df[feature_columns].values
    else:
        X = df.select_dtypes(include=[np.number]).values
    return X


def load_from_texts(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Convert unstructured text into embeddings using SentenceTransformers.
    If sentence-transformers is not available, raises RuntimeError.
    """
    if not SENTE:
        raise RuntimeError('sentence-transformers is not installed in the environment')
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True)
    return embs


def ingest_data(source_type: str, source: Optional[str] = None, payload: Optional[object] = None,
                feature_columns: Optional[List[str]] = None) -> np.ndarray:
    """Unified ingestion entrypoint.

    source_type: one of 'csv', 'sqlite', 'json', 'texts', or 'sample'
    source: path or DB connection string (for sqlite use file path)
    payload: used when source_type == 'json' (list of dicts) or 'texts' (list of str)
    """
    if source_type == 'csv':
        if not source:
            raise ValueError('source must be a path for csv')
        return load_from_csv(source, feature_columns=feature_columns)
    elif source_type == 'sqlite':
        if not source:
            raise ValueError('source must be a sqlite file path')
        query = payload if isinstance(payload, str) else 'SELECT * FROM data'
        return load_from_sqlite(source, query, feature_columns=feature_columns)
    elif source_type == 'json':
        if payload is None:
            raise ValueError('payload (list of dict) required for json source type')
        return load_from_json_payload(payload, feature_columns=feature_columns)
    elif source_type == 'texts':
        if payload is None:
            raise ValueError('payload (list of strings) required for texts source type')
        return load_from_texts(payload)
    elif source_type == 'sample':
        X, _ = load_sample_data()
        return X
    else:
        raise ValueError(f'Unsupported source_type: {source_type}')


def preprocess(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Scale numeric features using StandardScaler. Returns scaled X and the scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def pca_inferred_components(X: np.ndarray, variance_threshold: float = 0.90) -> int:
    """Compute PCA and return number of components needed to reach variance_threshold.

    This is used as a proxy for the intrinsic dimensionality of the data.
    """
    pca = PCA(n_components=min(X.shape[0], X.shape[1]))
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_comp = max(1, min(n_comp, X.shape[1]))
    return n_comp


def choose_k_from_candidates(pca_n_components: int, candidate_ks: List[int]) -> int:
    """Choose the candidate k that is closest to the pca_n_components.

    This is a simple heuristic — other methods (gap statistic, silhouette) may be
    used instead for more robust selection.
    """
    candidate_ks = sorted(candidate_ks)
    distances = [abs(k - pca_n_components) for k in candidate_ks]
    best_idx = int(np.argmin(distances))
    return candidate_ks[best_idx]


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """Run KMeans and return labels and fitted model."""
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def run_dbscan(X: np.ndarray, eps: float = 0.2, min_samples: int = 5) -> np.ndarray:
    """Run DBSCAN and return labels."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels


def generate_new_features(X: np.ndarray, labels: np.ndarray, pca_components: Optional[np.ndarray] = None) -> np.ndarray:
    """Append cluster labels (one-hot or integer) and optionally PCA components to features.

    Returns a new feature matrix with additional columns.
    """
    labels = np.asarray(labels).reshape(-1, 1)
    # Optionally convert labels to one-hot encoding (commented out by default)
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # labels_onehot = enc.fit_transform(labels)

    if pca_components is not None:
        return np.hstack([X, labels, pca_components])
    else:
        return np.hstack([X, labels])


def visualize_2d(X_scaled: np.ndarray, labels: np.ndarray, title: str = "Clustering"):
    """Simple 2-panel visualization for 2D data: scatter colored by cluster label."""
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=40)
    plt.title(title)
    plt.xlabel('PC1' if X_scaled.shape[1] >= 1 else 'x1')
    plt.ylabel('PC2' if X_scaled.shape[1] >= 2 else 'x2')
    plt.tight_layout()
    plt.show()


def run_pipeline(X: np.ndarray,
                 candidate_ks: List[int] = [6, 5, 4, 10],
                 pca_variance_threshold: float = 0.90,
                 clustering_method: str = 'kmeans'):
    """Run the full pipeline on input features X.

    Parameters
    - X: raw numeric features (n_samples, n_features)
    - candidate_ks: list of candidate cluster counts to choose from
    - pca_variance_threshold: how much variance PCA should explain to infer components
    - clustering_method: 'kmeans' or 'dbscan'
    """
    # 1) Preprocess
    X_scaled, scaler = preprocess(X)

    # 2) PCA to infer intrinsic components
    pca_n = pca_inferred_components(X_scaled, variance_threshold=pca_variance_threshold)
    print(f'PCA inferred components to reach {pca_variance_threshold*100:.0f}% variance: {pca_n}')

    # 3) Choose k from candidates
    chosen_k = choose_k_from_candidates(pca_n, candidate_ks)
    print(f'Candidate ks: {candidate_ks} -> chosen k: {chosen_k}')

    # 4) Perform clustering
    if clustering_method == 'kmeans':
        labels, model = run_kmeans(X_scaled, chosen_k)
    elif clustering_method == 'dbscan':
        labels = run_dbscan(X_scaled)
        model = None
    else:
        raise ValueError('Unsupported clustering_method')

    # 5) Compute PCA components for new features (optional)
    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    pca_components = pca.fit_transform(X_scaled)

    # 6) Generate new feature matrix
    X_new = generate_new_features(X_scaled, labels, pca_components)

    # 7) Visualize if the original data is 2D
    if X.shape[1] == 2:
        visualize_2d(X_scaled, labels, title=f'Clustering ({clustering_method}, k={chosen_k})')

    return {
        'scaler': scaler,
        'pca_n_components': pca_n,
        'chosen_k': chosen_k,
        'labels': labels,
        'model': model,
        'X_new': X_new
    }


if __name__ == '__main__':
    # Demo run using synthetic data
    X, y = load_sample_data()
    result = run_pipeline(X, candidate_ks=[6, 5, 4, 10], pca_variance_threshold=0.90, clustering_method='kmeans')
    print('Pipeline finished. New feature matrix shape:', result['X_new'].shape)
