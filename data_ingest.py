"""Data ingestion helpers and loader registry.

Design goals
- centralize ingestion logic (CSV, SQLite, JSON payloads, text->embeddings)
- provide a simple registry so additional loaders (S3, Postgres, Kafka, YAML synthetic) can be added
- keep the public API small: DataIngestor.ingest(source_type, source=..., payload=..., feature_columns=...)

Best practices and library notes
- pandas: used for convenient, robust CSV/SQL/JSON -> DataFrame conversions. For very large files prefer
  chunked reading (pd.read_csv(..., chunksize=...)) or a streaming approach rather than loading everything
  into memory. Consider using pyarrow or Dask for out-of-core workflows.
- numpy: preferred numeric array representation returned by ingestion. This keeps the downstream ML pipeline
  independent of pandas. If you need the DataFrame (with dtypes and non-numeric columns), pass return_df=True.
- sentence-transformers: convenient for quick text embeddings. Loading models is expensive; reuse a single
  model instance where possible. We cache the model on first use to avoid repeated loads.
- YAML (PyYAML): supported in the project for synthetic schema-driven generation. If unavailable, the YAML
  synthetic loader is skipped and a clear error is raised.

On generating synthetic data via distributions
- The ingestion layer itself is typically not responsible for data generation. Instead, use a separate
  SyntheticDataGenerator (see `synthetic_data.py`) that lives alongside `DataIngestor` and registers a
  loader (e.g., 'synthetic' or 'yaml') which calls the generator. This keeps responsibilities clear:
    * SyntheticDataGenerator: focuses on generating rows according to a schema/distribution
    * DataIngestor: focuses on converting sources/payloads into arrays/dataframes for downstream steps
- It's reasonable to allow `DataIngestor` to offer a convenience loader for synthetic data (it simply
  delegates to the generator). It's best practice to keep generation deterministic when needed via
  a provided random seed.

Usage example
  from data_ingest import DataIngestor
  di = DataIngestor()
  X = di.ingest('csv', source='data/myfile.csv')
  X = di.ingest('json', payload=[{'a':1,'b':2}, ...])
  X = di.ingest('texts', payload=['hello world', 'another doc'])  # requires sentence-transformers

"""

from typing import Callable, Dict, List, Optional, Tuple
import logging
import os
import numpy as np
import pandas as pd
import sqlite3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# sentence-transformers model caching
_SENTE_AVAILABLE = False
_SENTE_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    _SENTE_AVAILABLE = True
except Exception:
    _SENTE_AVAILABLE = False

# optional synthetic_data integration (if present we'll register a loader)
_SYNTHETIC_AVAILABLE = False
try:
    import synthetic_data as _synth
    _SYNTHETIC_AVAILABLE = True
except Exception:
    _SYNTHETIC_AVAILABLE = False


class DataIngestor:
    """Pluggable data ingestion class.

    Notes on return values and types:
      - By default `ingest` returns a numpy ndarray with numeric columns only.
      - Pass `feature_columns` to select columns explicitly (list of column names).
      - If you need the raw pandas DataFrame (with dates / categories), use `return_df=True`.

    The loader function signature is: loader(source, payload, feature_columns, return_df) -> (X, df)
    where X is a numpy array (or None) and df is the DataFrame (or None). The public API returns X
    unless return_df=True in which case the DataFrame is returned.
    """

    def __init__(self):
        # registry mapping source_type to loader function
        self.loaders: Dict[str, Callable] = {
            'csv': self._load_from_csv,
            'sqlite': self._load_from_sqlite,
            'json': self._load_from_json_payload,
            'texts': self._load_from_texts,
            'sample': self._load_sample_data,
        }

        # register synthetic/yaml loader if synthetic_data is available
        if _SYNTHETIC_AVAILABLE:
            self.register_loader('yaml', self._load_from_yaml)
            self.register_loader('synthetic', self._load_from_synthetic)
            logger.info('Registered synthetic/yaml loaders from synthetic_data')

    def register_loader(self, name: str, func: Callable):
        """Register a custom loader function under `name`.

        The function must accept (source, payload, feature_columns, return_df) and return
        a tuple (X, df) where X is a numpy array or None and df is a pandas DataFrame or None.
        """
        self.loaders[name] = func

    def ingest(self, source_type: str, source: Optional[str] = None, payload: Optional[object] = None,
               feature_columns: Optional[List[str]] = None, return_df: bool = False) -> object:
        """Unified entrypoint to ingest data from many sources.

        Parameters
        - source_type: registered loader name (e.g., 'csv', 'sqlite', 'json', 'texts', 'sample', 'yaml')
        - source: file path, DB path, or other source identifier (optional)
        - payload: used for json (list[dict]), texts (list[str]), or SQL query (string)
        - feature_columns: optional list of columns to select (applies to DataFrame loaders)
        - return_df: if True return the pandas DataFrame instead of numpy array

        Returns
        - numpy.ndarray (n_samples, n_features) by default, or pandas.DataFrame if return_df=True
        """
        loader = self.loaders.get(source_type)
        if loader is None:
            raise ValueError(f'Unsupported source_type: {source_type}')

        X, df = loader(source, payload, feature_columns, return_df)
        if return_df:
            return df
        return X

    # ----- built-in loaders -----
    def _load_sample_data(self, source=None, payload=None, feature_columns=None, return_df=False) -> Tuple[np.ndarray, pd.DataFrame]:
        """Small convenience loader for quick demos.

        By default returns the 2D circle dataset (useful for visualization). If you prefer a
        different demo dataset, replace this method or register a new loader (e.g., 'blobs').
        """
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
        df = pd.DataFrame(X, columns=['x0', 'x1'])
        if return_df:
            return None, df
        return X, None

    def _load_from_csv(self, source: str, payload=None, feature_columns: Optional[List[str]] = None, return_df=False) -> Tuple[np.ndarray, pd.DataFrame]:
        if not source:
            raise ValueError('CSV loader requires a file path in `source`')
        if not os.path.exists(source):
            raise FileNotFoundError(f'CSV file not found: {source}')
        # Small convenience: let pandas infer dtypes; for production pass dtype hints
        df = pd.read_csv(source)
        if feature_columns:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f'Requested feature_columns not found in CSV: {missing}')
            if return_df:
                return None, df[feature_columns]
            return df[feature_columns].values, None

        # default: select numeric dtypes
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            logger.warning('CSV has no numeric columns; returning empty array')
            if return_df:
                return None, df
            return np.zeros((len(df), 0)), None
        if return_df:
            return None, num_df
        return num_df.values, None

    def _load_from_sqlite(self, source: str, payload=None, feature_columns: Optional[List[str]] = None, return_df=False) -> Tuple[np.ndarray, pd.DataFrame]:
        if not source:
            raise ValueError('SQLite loader requires a DB file path in `source`')
        if not os.path.exists(source):
            raise FileNotFoundError(f'SQLite DB not found: {source}')
        query = payload if isinstance(payload, str) else 'SELECT * FROM data'
        conn = sqlite3.connect(source)
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
        if feature_columns:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f'Requested feature_columns not found in SQLite result: {missing}')
            if return_df:
                return None, df[feature_columns]
            return df[feature_columns].values, None

        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            logger.warning('SQLite result has no numeric columns; returning empty array')
            if return_df:
                return None, df
            return np.zeros((len(df), 0)), None
        if return_df:
            return None, num_df
        return num_df.values, None

    def _load_from_json_payload(self, source=None, payload: List[dict] = None, feature_columns: Optional[List[str]] = None, return_df=False) -> Tuple[np.ndarray, pd.DataFrame]:
        if payload is None:
            raise ValueError('JSON loader requires payload (list of dict)')
        if not isinstance(payload, (list, tuple)):
            raise ValueError('JSON payload must be a list of dicts')
        df = pd.DataFrame(payload)
        if feature_columns:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f'Requested feature_columns not found in JSON payload: {missing}')
            if return_df:
                return None, df[feature_columns]
            return df[feature_columns].values, None

        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            logger.warning('JSON payload has no numeric columns; returning empty array')
            if return_df:
                return None, df
            return np.zeros((len(df), 0)), None
        if return_df:
            return None, num_df
        return num_df.values, None

    def _load_from_texts(self, source=None, payload: List[str] = None, feature_columns: Optional[List[str]] = None, return_df=False) -> Tuple[np.ndarray, pd.DataFrame]:
        if payload is None:
            raise ValueError('Texts loader requires payload (list of strings)')
        if not _SENTE_AVAILABLE:
            raise RuntimeError('sentence-transformers is not installed in the environment')
        global _SENTE_MODEL
        if _SENTE_MODEL is None:
            # lazy-load and cache model for reuse
            model_name = os.environ.get('SENTE_MODEL', 'all-MiniLM-L6-v2')
            logger.info('Loading SentenceTransformer model: %s', model_name)
            _SENTE_MODEL = SentenceTransformer(model_name)
        # encode -> returns numpy array
        embs = _SENTE_MODEL.encode(payload, convert_to_numpy=True)
        if return_df:
            # return DataFrame with columns emb_0..emb_n
            cols = [f'emb_{i}' for i in range(embs.shape[1])]
            return None, pd.DataFrame(embs, columns=cols)
        return embs, None

    # ---- synthetic loaders delegate to synthetic_data module if available ----
    def _load_from_synthetic(self, source=None, payload=None, feature_columns: Optional[List[str]] = None, return_df=False):
        if not _SYNTHETIC_AVAILABLE:
            raise RuntimeError('synthetic_data module not available; install or add it to PYTHONPATH')
        # payload can be schema dict or None to use defaults
        if isinstance(payload, dict):
            df = _synth.generate_from_schema(payload)
        elif isinstance(source, str) and os.path.exists(source):
            df = _synth.generate_from_yaml(source)
        else:
            # fallback: use default small sample schema
            df = _synth.generate_from_schema({'n': 100, 'fields': {col: spec for col, spec in _synth.__dict__.items() if False}})
        if feature_columns:
            return (df[feature_columns].values, None) if not return_df else (None, df[feature_columns])
        num_df = df.select_dtypes(include=[np.number])
        return (num_df.values, None) if not return_df else (None, num_df)

    def _load_from_yaml(self, source=None, payload=None, feature_columns: Optional[List[str]] = None, return_df=False):
        if not _SYNTHETIC_AVAILABLE:
            raise RuntimeError('synthetic_data module not available; install or add it to PYTHONPATH')
        if not source or not os.path.exists(source):
            raise ValueError('YAML loader requires a valid file path in `source`')
        df = _synth.generate_from_yaml(source)
        if feature_columns:
            return (df[feature_columns].values, None) if not return_df else (None, df[feature_columns])
        num_df = df.select_dtypes(include=[np.number])
        return (num_df.values, None) if not return_df else (None, num_df)


if __name__ == '__main__':
    # quick smoke test
    di = DataIngestor()
    X = di.ingest('sample')
    print('Loaded sample X shape:', X.shape)
