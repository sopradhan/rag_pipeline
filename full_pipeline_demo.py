"""full_pipeline_demo.py
Demonstrates complete workflow:
1. Generate synthetic data with missing values
2. Ingest & preprocess (imputation, scaling)
3. Dimension reduction (PCA)
4. Clustering (trying multiple k values)
5. Feature engineering from clusters
6. Export enhanced dataset
"""
import os
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import davies_bouldin_score, silhouette_samples

from synthetic_data import generate_from_schema, generate_from_yaml
from data_ingest import DataIngestor

# Optional imputers/clusterers
try:
    from sklearn.impute import KNNImputer
except Exception:
    KNNImputer = None

try:
    # IterativeImputer is experimental in some sklearn versions
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
except Exception:
    IterativeImputer = None

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
except Exception:
    DBSCAN = None
    AgglomerativeClustering = None


def load_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_imputer(name: str):
    name = (name or 'median').lower()
    if name == 'median':
        return SimpleImputer(strategy='median')
    if name == 'mean':
        return SimpleImputer(strategy='mean')
    if name == 'most_frequent':
        return SimpleImputer(strategy='most_frequent')
    if name == 'knn' and KNNImputer is not None:
        return KNNImputer()
    if name == 'iterative' and IterativeImputer is not None:
        return IterativeImputer()
    raise ValueError(f'Unknown or unavailable imputer: {name}')


def get_scaler(name: str):
    name = (name or 'robust').lower()
    if name == 'robust':
        return RobustScaler()
    if name == 'standard':
        return StandardScaler()
    if name == 'minmax':
        return MinMaxScaler()
    if name == 'none':
        return None
    raise ValueError(f'Unknown scaler: {name}')

# Synthetic data generation is provided by synthetic_data.py (generate_from_schema / generate_from_yaml)
# The in-file generator was removed to keep the pipeline modular — the main() function uses
# generate_from_yaml or generate_from_schema from the external module.

def preprocess_data(df, numeric_cols=None):
    """Preprocess numeric features with imputation and scaling"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Default pipeline: median imputation + robust scaling
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df[numeric_cols])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return pd.DataFrame(X_scaled, columns=numeric_cols, index=df.index)

def reduce_dimensions(X, n_components=None, variance_threshold=0.95):
    """Apply PCA with either explicit components or variance threshold"""
    if n_components is None:
        # Find components explaining variance_threshold of variance
        pca = PCA()
        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Plot explained variance
    plt.figure(figsize=(10, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Components')
    plt.savefig('pca_variance.png')
    plt.close()
    
    return X_reduced, pca

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    K = range(2, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 4))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    optimal_k = K[np.argmax(silhouette_scores)]
    return optimal_k

def engineer_features(X, cluster_labels, original_df):
    """Generate new features based on clustering results"""
    # Create enhanced dataset
    enhanced = pd.DataFrame()
    
    # Original features
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    enhanced[numeric_cols] = original_df[numeric_cols]
    
    # Categorical features (one-hot encoded)
    cat_cols = original_df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        enhanced = pd.concat([enhanced, pd.get_dummies(original_df[cat_cols], prefix=cat_cols)], axis=1)
    
    # Add cluster assignment
    enhanced['cluster'] = cluster_labels
    
    # Add distance to cluster center
    kmeans = KMeans(n_clusters=len(np.unique(cluster_labels)), random_state=42)
    kmeans.fit(X)
    distances = kmeans.transform(X)
    enhanced['distance_to_center'] = distances.min(axis=1)
    
    # Add top 2 PCA components if X was PCA-transformed
    if X.shape[1] >= 2:
        enhanced['pca1'] = X[:, 0]
        enhanced['pca2'] = X[:, 1]
    
    return enhanced


def plot_dendrogram(X, labels=None, method='ward', save_path='dendrogram.png'):
    """Plot dendrogram. If `labels` (cluster labels per sample) is provided, color leaf labels by cluster."""
    Z = linkage(X, method=method)
    plt.figure(figsize=(14, 6))
    ddata = dendrogram(Z, truncate_mode='level', p=5, color_threshold=None)
    ax = plt.gca()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')

    # If cluster labels provided, color the x tick labels according to cluster membership
    if labels is not None and len(labels) == X.shape[0]:
        try:
            leaves = ddata.get('leaves', [])
            ticks = ax.get_xmajorticklabels()
            # build a colormap for clusters
            unique = np.unique(labels)
            cmap = plt.get_cmap('tab10')
            color_map = {u: cmap(i % 10) for i, u in enumerate(unique)}
            for idx, leaf in enumerate(leaves):
                if idx < len(ticks):
                    lbl = ticks[idx]
                    cluster_id = labels[leaf]
                    lbl.set_color(color_map.get(cluster_id, 'k'))
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_silhouette(X, labels, save_path='silhouette.png'):
    from sklearn.metrics import silhouette_samples
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        ith_vals = np.sort(sil_vals[labels == i])
        size = ith_vals.shape[0]
        y_upper = y_lower + size
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Cluster')
    plt.title('Silhouette Plot per Cluster')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cluster_mountain(X_reduced, labels, df=None, numeric_cols=None, save_path='cluster_mountain.png'):
    """Create a mountain-style cluster visualization.

    - Plots cluster centroids in the first two PCA dims (or dims 0/1 if present)
    - Marker size ~ cluster size
    - Color by average silhouette per cluster
    - Saves image to save_path
    """
    import matplotlib.cm as cm
    from collections import Counter
    from scipy.spatial import ConvexHull
    # Ensure at least 2 dims
    if X_reduced.shape[1] < 2:
        X_plot = np.hstack([X_reduced, np.zeros((X_reduced.shape[0], 1))])
    else:
        X_plot = X_reduced[:, :2]

    unique = np.unique(labels)
    counts = Counter(labels)
    # compute centroids
    centroids = np.array([X_plot[labels == u].mean(axis=0) for u in unique])
    # silhouette per sample
    try:
        sil_vals = silhouette_samples(X_reduced, labels)
    except Exception:
        sil_vals = np.zeros(len(labels))
    sil_by_cluster = {u: float(np.nanmean(sil_vals[labels == u])) if np.any(labels == u) else 0.0 for u in unique}
    sizes = np.array([counts[u] for u in unique])

    # Normalize sizes for plotting
    size_norm = 150 * (sizes / sizes.max()) + 80
    # Color map by silhouette
    sil_vals_centroid = np.array([sil_by_cluster[u] for u in unique])
    cmap = cm.get_cmap('viridis')

    plt.figure(figsize=(10, 6))
    # draw convex hulls for clusters (if > 2 points)
    for i, u in enumerate(unique):
        pts = X_plot[labels == u]
        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                plt.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.12, color=cmap(sil_by_cluster[u]))
            except Exception:
                pass

    sc = plt.scatter(centroids[:, 0], centroids[:, 1], s=size_norm, c=sil_vals_centroid, cmap=cmap, edgecolor='k', alpha=0.95)
    for i, u in enumerate(unique):
        plt.text(centroids[i, 0], centroids[i, 1], str(u), fontsize=9, ha='center', va='center', weight='bold')

    # annotate top features per cluster if df and numeric_cols provided
    if df is not None and numeric_cols:
        for i, u in enumerate(unique):
            idx = np.where(labels == u)[0]
            if len(idx) == 0:
                continue
            means = df.iloc[idx][numeric_cols].mean()
            # pick top 1 feature dev from global mean
            global_mean = df[numeric_cols].mean()
            diffs = (means - global_mean).abs().sort_values(ascending=False)
            top = diffs.index[0]
            txt = f"{top}: {means[top]:.1f}"
            # place label slightly offset from centroid
            plt.text(centroids[i, 0] + 0.02, centroids[i, 1] + 0.02, txt, fontsize=8)

    plt.colorbar(sc, label='Average silhouette')
    plt.title('Cluster Mountain: centroids sized by cluster size, colored by silhouette')
    plt.xlabel('PCA dim 1')
    plt.ylabel('PCA dim 2')

    # add inset bar chart for cluster sizes and silhouette
    try:
        from matplotlib.transforms import Bbox
        ax = plt.gca()
        left, bottom, width, height = 0.68, 0.55, 0.28, 0.35
        axins = plt.axes([left, bottom, width, height])
        order = np.argsort(sizes)[::-1]
        labels_order = [str(unique[i]) for i in order]
        size_vals = sizes[order]
        sil_vals_order = sil_vals_centroid[order]
        axins.barh(labels_order, size_vals, color='gray', alpha=0.6)
        axins2 = axins.twiny()
        axins2.plot(sil_vals_order, range(len(sil_vals_order)), 'o-', color='C1')
        axins.set_title('Cluster sizes (bars) & silhouette (line)')
        axins.invert_yaxis()
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def explain_clusters_llm(df, numeric_cols, X_reduced, labels, score_df, out_path='cluster_explanation.txt'):
    """Assemble cluster diagnostics and ask the LLM to explain: Why / How / When / Next Steps.

    Writes the textual explanation to out_path.
    """
    try:
        from llm_agent import summarize_insights
    except Exception:
        summarize_insights = None

    # Build diagnostics
    clusters = np.unique(labels)
    rows = []
    try:
        sil_vals = silhouette_samples(X_reduced, labels)
    except Exception:
        sil_vals = np.zeros(len(labels))

    # compute centroids and avg intra-cluster distance
    centroids = {int(c): X_reduced[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(X_reduced.shape[1]) for c in clusters}
    for c in clusters:
        idx = np.where(labels == c)[0]
        size = len(idx)
        sil_avg = float(np.nanmean(sil_vals[idx])) if size > 0 else 0.0
        means = df.iloc[idx][numeric_cols].mean().to_dict() if size > 0 else {k: None for k in numeric_cols}
        # avg distance to centroid
        if size > 0:
            dists = np.linalg.norm(X_reduced[idx] - centroids[int(c)], axis=1)
            avg_dist = float(np.nanmean(dists))
        else:
            avg_dist = float('nan')
        rows.append({'cluster': int(c), 'size': int(size), 'silhouette': sil_avg, 'avg_dist': avg_dist, 'means': means})

    # Decide best cluster(s) heuristically for deterministic fallback
    try:
        best_by_sil = max(rows, key=lambda r: (r['silhouette'], r['size']))
    except Exception:
        best_by_sil = None

    # Create analysis blob for the LLM
    lines = []
    lines.append('Cluster diagnostics:')
    for r in rows:
        lines.append(f"Cluster {r['cluster']}: size={r['size']}, silhouette={r['silhouette']:.3f}, avg_dist={r['avg_dist']:.3f}")
        top_feats = ', '.join([f"{k}={v:.2f}" for k, v in sorted(r['means'].items(), key=lambda x: -abs(x[1]) )[:3]]) if r['means'] else ''
        if top_feats:
            lines.append(f"  Top feature means: {top_feats}")

    # Add cluster score sweep for context
    lines.append('\nCluster score sweep (top rows):')
    try:
        lines.append(score_df.sort_values('combined', ascending=False).to_csv(index=False))
    except Exception:
        lines.append(str(score_df))

    prompt = '\n'.join(lines)

    instruction = (
        "You are a senior data scientist. Given the diagnostics below and the cluster score sweep, do the following succinctly:\n"
        "1) Interpret the cluster scores and say which k appears best and why.\n"
        "2) For the final clustering (provided per-cluster diagnostics), identify which cluster(s) are higher-quality (stable/cohesive) and why.\n"
        "3) Provide 3 concise next steps to operationalize the clusters (monitoring, thresholds, feature checks).\n"
        "Return plain text with short bullet sections: SUMMARY (1-2 lines), INTERPRETATION, BEST CLUSTERS, NEXT STEPS."
    )

    full_prompt = instruction + '\n\nDiagnostics:\n' + prompt

    if summarize_insights:
        try:
            out = summarize_insights(full_prompt)
        except Exception:
            out = 'Cluster explanation fallback (LLM failed).\n' + '\n'.join(lines)
    else:
        # Deterministic fallback: summarize scores and pick best cluster by silhouette (tie-breaker size)
        fallback_lines = []
        fallback_lines.append('Cluster explanation (LLM unavailable):')
        # best k from score_df
        try:
            best_k_row = score_df.sort_values('combined', ascending=False).iloc[0]
            fallback_lines.append(f"SUMMARY: Best k according to combined metric is {int(best_k_row['k'])} (silhouette={best_k_row['silhouette']:.3f}, db={best_k_row['davies_bouldin']:.3f})")
        except Exception:
            fallback_lines.append('SUMMARY: Could not determine best k from score sweep')

        if best_by_sil:
            fallback_lines.append(f"INTERPRETATION: Cluster {best_by_sil['cluster']} looks best by silhouette ({best_by_sil['silhouette']:.3f}) and size={best_by_sil['size']}.")
            fallback_lines.append(f"DETAILS: avg intra-cluster distance={best_by_sil['avg_dist']:.3f}; top features: " + ', '.join([f"{k}={v:.2f}" for k, v in list(best_by_sil['means'].items())[:3]]))
        else:
            fallback_lines.append('INTERPRETATION: No clear best cluster (insufficient data).')

        fallback_lines.append('NEXT STEPS: 1) Validate clusters on hold-out / temporal split. 2) Monitor silhouette per cluster and sizes. 3) Investigate top features to create alerts/thresholds.')
        out = '\n'.join(fallback_lines)

    # If the LLM output did not explicitly name a best cluster or interpret scores, append a deterministic summary
    try:
        lower_out = out.lower()
    except Exception:
        lower_out = ''
    if 'best' not in lower_out and best_by_sil is not None:
        append = []
        append.append('\nDETERMINISTIC SUMMARY:')
        try:
            best_k_row = score_df.sort_values('combined', ascending=False).iloc[0]
            append.append(f"Best k by combined metric: {int(best_k_row['k'])} (sil={best_k_row['silhouette']:.3f}, db={best_k_row['davies_bouldin']:.3f})")
        except Exception:
            pass
        append.append(f"Best cluster by silhouette: Cluster {best_by_sil['cluster']} (silhouette={best_by_sil['silhouette']:.3f}, size={best_by_sil['size']})")
        append.append('Suggested next steps: validate clusters on hold-out data; monitor silhouette and cluster sizes; investigate top features and set thresholds/alerts.')
        out = out + '\n' + '\n'.join(append)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print('Wrote cluster explanation to', out_path)


def auto_select_k(X, k_min=2, k_max=12):
    """Search K by silhouette score and Davies-Bouldin (higher silhouette, lower DB)."""
    best_k = None
    best_score = -999
    scores = []
    for k in range(k_min, min(k_max, X.shape[0]-1) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
        except Exception:
            sil = -1
            db = np.inf
        # combine into a single metric: silhouette - normalized db
        score = sil - (db / 10.0)
        scores.append((k, sil, db, score))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores


def _load_provider_from_string(path: str):
    """Load a provider callable/class from a dotted path string like 'module.Class' or 'module:Class'.

    Returns a callable/provider instance or raises ImportError.
    """
    import importlib
    if ':' in path:
        mod_path, cls_name = path.split(':', 1)
    elif '.' in path:
        parts = path.split('.')
        mod_path = '.'.join(parts[:-1])
        cls_name = parts[-1]
    else:
        raise ImportError(f'Invalid provider path: {path}')
    mod = importlib.import_module(mod_path)
    provider = getattr(mod, cls_name)
    return provider


def run_pipeline_on_raw(raw_csv: str,
                        plants=None,
                        substations=None,
                        features=None,
                        imputer_name='median',
                        scaler_name='robust',
                        max_k=12,
                        metadata_provider=None):
    """Ingest a raw CSV with plant/substation metadata, select numeric features, run PCA & clustering.

    Steps:
    - Load CSV
    - Remove timestamp and categorical columns from PCA inputs
    - Impute & scale
    - PCA (plot scree)
    - Auto-select k and cluster
    - Generate dendrogram and silhouette plots
    - Merge labels and PCA metadata back into original DF and save
    """
    print('Loading raw CSV:', raw_csv)
    df = pd.read_csv(raw_csv)

    # Metadata enrichment (pluggable)
    # If `metadata_provider` is provided it should be either:
    # - a callable that accepts the DataFrame and returns the DataFrame (possibly enriched), or
    # - a string import path to a provider class/function (e.g. 'mymodule:Provider') which will be called
    # If metadata_provider is not provided, the older `plants`/`substations` lists can still be used
    # but this is deprecated — prefer providing a provider that knows your schema.
    if metadata_provider:
        # load provider if string
        provider_callable = None
        if isinstance(metadata_provider, str):
            try:
                provider = _load_provider_from_string(metadata_provider)
                # If provider is a class, instantiate without args; else use as callable
                if isinstance(provider, type):
                    provider_callable = provider()
                else:
                    provider_callable = provider
            except Exception as e:
                raise RuntimeError(f'Unable to load metadata_provider {metadata_provider}: {e}')
        elif callable(metadata_provider):
            provider_callable = metadata_provider
        else:
            raise ValueError('metadata_provider must be a callable or import path string')

        # call provider and expect a DataFrame in return
        try:
            df = provider_callable(df)
        except Exception as e:
            raise RuntimeError(f'Metadata provider call failed: {e}')
    else:
        # backward compatible behavior (deprecated): populate missing plant/substation only when provided
        if plants and 'plant' not in df.columns:
            print('Deprecation notice: passing `plants` to run_pipeline_on_raw is deprecated; please use metadata_provider.')
            df['plant'] = np.random.choice(plants, size=len(df))
        if substations and 'substation' not in df.columns:
            print('Deprecation notice: passing `substations` to run_pipeline_on_raw is deprecated; please use metadata_provider.')
            df['substation'] = np.random.choice(substations, size=len(df))

    # Remove timestamp-like columns
    drop_like = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    df_pca = df.drop(columns=drop_like, errors='ignore')

    # Keep only numeric features for PCA (or explicit features if provided)
    if features:
        numeric_cols = [f for f in features if f in df_pca.columns]
    else:
        # Start with numeric dtypes
        numeric_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude any columns that are actually categorical/object in the original df (defensive)
        cat_cols = df_pca.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in cat_cols]
        # Exclude likely identifier columns (id, name, station, substation, plant) even if numeric
        id_like = [c for c in df_pca.columns if any(p in c.lower() for p in ('id', 'name', 'station', 'substation', 'plant'))]
        numeric_cols = [c for c in numeric_cols if c not in id_like]

    if not numeric_cols:
        raise ValueError('No numeric columns found for PCA')

    print('Using numeric columns for PCA:', numeric_cols)

    # Impute & scale
    imputer = get_imputer(imputer_name)
    scaler = get_scaler(scaler_name)
    X_imputed = imputer.fit_transform(df[numeric_cols])
    X_scaled = scaler.fit_transform(X_imputed)

    # PCA: keep components explaining 95% or at most number of features
    X_reduced, pca = reduce_dimensions(X_scaled, variance_threshold=0.95)
    print('PCA reduced shape:', X_reduced.shape)

    # Auto-select K
    k_opt, scores = auto_select_k(X_reduced, k_min=2, k_max=max_k)
    print('Auto-selected k:', k_opt)

    # Final clustering
    kmeans = KMeans(n_clusters=k_opt, random_state=42)
    labels = kmeans.fit_predict(X_reduced)

    # Plots
    plot_dendrogram(X_scaled, save_path='dendrogram.png')
    plot_silhouette(X_reduced, labels, save_path='silhouette_plot.png')

    # Additional cluster visualization: mountain-style (centroids + size/silhouette)
    try:
        fn = globals().get('plot_cluster_mountain')
        if fn:
            fn(X_reduced, labels, df=df, numeric_cols=numeric_cols, save_path='cluster_mountain.png')
        else:
            print('plot_cluster_mountain not available in globals()')
    except Exception as e:
        print('Failed to create cluster mountain plot:', e)

    # Ask LLM to explain clusters (Why / How / When / Next Steps)
    try:
        # Prepare a score DataFrame for LLM context
        score_df = pd.DataFrame(scores, columns=['k', 'silhouette', 'davies_bouldin', 'combined'])
        fn2 = globals().get('explain_clusters_llm')
        if fn2:
            fn2(df, numeric_cols, X_reduced, labels, score_df, out_path='cluster_explanation.txt')
        else:
            print('explain_clusters_llm not available in globals()')
    except Exception as e:
        print('LLM cluster explanation failed:', e)

    # Merge labels and PCA metadata into original DF
    merged = df.copy()
    merged['cluster'] = labels
    # PCA components
    for i in range(min(2, X_reduced.shape[1])):
        merged[f'pca_{i+1}'] = X_reduced[:, i]

    out_csv = 'enhanced_clustered.csv'
    merged.to_csv(out_csv, index=False)
    print('Wrote clustered enhanced dataset to', out_csv)

    # Summarize scores
    score_df = pd.DataFrame(scores, columns=['k', 'silhouette', 'davies_bouldin', 'combined'])
    score_df.to_csv('cluster_scores.csv', index=False)
    print('Saved cluster score sweep to cluster_scores.csv')

    # Ask LLM for suggestions
    try:
        from llm_agent import summarize_insights
        analysis_blob = f'PCA explained variance: {pca.explained_variance_ratio_}\nBest k: {k_opt}\nScores:\n{score_df.head().to_csv()}'
        print('Requesting LLM suggestions...')
        llm_out = summarize_insights(analysis_blob)
        print('\nLLM Suggestions:\n', llm_out)
        with open('analysis_llm_summary.txt', 'w', encoding='utf-8') as f:
            f.write(llm_out)
    except Exception as e:
        print('LLM not available or failed:', e)

    return merged


def hyperparam_optimization(raw_csv: str, iterations: int = 5, max_k: int = 12):
    """Optimize hyperparameters by asking the LLM for suggestions and evaluating them.

    For each iteration:
    - Build a short analysis blob from the data (missingness, PCA explained variance)
    - Ask the LLM for JSON hyperparams via llm_agent.suggest_hyperparams
    - Evaluate the suggested config (imputer, scaler, k) by computing silhouette and DB
    - Track the best config using a combined metric (sil - db/10)

    After iterations, re-run the full pipeline with the best config and save outputs.
    """
    print(f"Starting hyperparameter optimization for {raw_csv} (iterations={iterations})")
    df = pd.read_csv(raw_csv)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError('No numeric columns found for hyperparameter optimization')

    # Initial analysis blob (missingness + feature count)
    miss = df[numeric_cols].isnull().mean() * 100.0
    miss_summary = ", ".join([f"{c}:{miss[c]:.2f}%" for c in numeric_cols[:5]])

    # PCA with default imputer/scaler to provide initial context
    imputer0 = SimpleImputer(strategy='median')
    scaler0 = RobustScaler()
    X0 = scaler0.fit_transform(imputer0.fit_transform(df[numeric_cols]))
    X0_reduced, pca0 = reduce_dimensions(X0, variance_threshold=0.95)
    analysis_blob_base = f"{len(numeric_cols)} features. Missing% (first cols): {miss_summary}. PCA explained variance: {pca0.explained_variance_ratio_[:4].tolist()}"

    try:
        from llm_agent import suggest_hyperparams
    except Exception:
        suggest_hyperparams = None

    best_config = None
    best_score = -999
    history = []

    for it in range(1, iterations + 1):
        print(f"\nIteration {it}/{iterations}")
        # Ask LLM (or fallback) for suggestion
        prompt = analysis_blob_base + f"\nPrevious best score: {best_score}\nIteration: {it}"
        if suggest_hyperparams:
            try:
                suggestion = suggest_hyperparams(prompt)
            except Exception as e:
                print('LLM suggestion failed, using heuristic fallback:', e)
                suggestion = None
        else:
            suggestion = None

        if not suggestion or not isinstance(suggestion, dict):
            # fallback deterministic heuristic
            suggestion = {'k': min(max(3, len(numeric_cols) * 2), max_k), 'imputer': 'median', 'scaler': 'robust'}

        # sanitize suggestion
        k_sugg = int(suggestion.get('k', 3))
        k_sugg = max(2, min(k_sugg, max_k))
        imputer_name = suggestion.get('imputer', 'median')
        scaler_name = suggestion.get('scaler', 'robust')
        print(f"Suggested config: k={k_sugg}, imputer={imputer_name}, scaler={scaler_name}")

        # Evaluate suggestion
        try:
            imp = get_imputer(imputer_name)
            scl = get_scaler(scaler_name)
            X_imp = imp.fit_transform(df[numeric_cols])
            X_scl = scl.fit_transform(X_imp) if scl is not None else X_imp
            X_red, _ = reduce_dimensions(X_scl, variance_threshold=0.95)
            kmeans = KMeans(n_clusters=k_sugg, random_state=42)
            labels = kmeans.fit_predict(X_red)
            sil = silhouette_score(X_red, labels)
            db = davies_bouldin_score(X_red, labels)
            combined = sil - (db / 10.0)
        except Exception as e:
            print('Evaluation failed for suggestion:', e)
            sil, db, combined = -1.0, float('inf'), -999

        print(f"Evaluation -> silhouette: {sil:.4f}, davies_bouldin: {db:.4f}, combined: {combined:.4f}")
        history.append({'iteration': it, 'k': k_sugg, 'imputer': imputer_name, 'scaler': scaler_name, 'silhouette': sil, 'db': db, 'combined': combined})

        if combined > best_score:
            best_score = combined
            best_config = {'k': k_sugg, 'imputer': imputer_name, 'scaler': scaler_name}

    # Persist history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv('hyperopt_history.csv', index=False)
    print('\nHyperopt history saved to hyperopt_history.csv')

    if best_config is None:
        raise RuntimeError('Hyperparameter optimization failed to find any valid config')

    print('\nBest config found:', best_config, 'score:', best_score)
    # Re-run full pipeline with best config and save outputs
    merged = run_pipeline_on_raw(raw_csv,
                                plants=None,
                                substations=None,
                                features=None,
                                imputer_name=best_config['imputer'],
                                scaler_name=best_config['scaler'],
                                max_k=best_config['k'])
    print('Final pipeline completed with best hyperparameters')
    return best_config, hist_df

def main():
    # parse CLI args and config at start of main
    parser = argparse.ArgumentParser(description='Full pipeline demo: synthetic -> ingest -> preprocess -> PCA -> cluster -> features')
    parser.add_argument('--config', '-c', default='pipeline_config.yaml', help='Path to pipeline YAML config')
    parser.add_argument('--n-samples', type=int, dest='n_samples', help='Number of synthetic samples to generate')
    parser.add_argument('--imputer', default='median', help='Imputer to use: median|mean|most_frequent|knn|iterative')
    parser.add_argument('--scaler', default='robust', help='Scaler: robust|standard|minmax|none')
    parser.add_argument('--max-clusters', type=int, default=10, help='Max clusters to search for')
    parser.add_argument('--cluster-algo', default='kmeans', help='Clustering algorithm: kmeans|dbscan|agglomerative')
    parser.add_argument('--schema', default=None, help='Path to YAML schema to use for generation (overrides built-in schema)')
    parser.add_argument('--hyperopt', action='store_true', help='Run LLM-driven hyperparameter optimization instead of default single run')
    parser.add_argument('--hyperopt-iter', type=int, default=5, help='Number of hyperopt iterations (LLM suggestions)')
    parsed = parser.parse_args()

    # load config and merge
    cfg = load_config(parsed.config)
    # CLI overrides
    args = parsed
    if args.n_samples:
        cfg['n_samples'] = args.n_samples
    if args.imputer:
        cfg['imputer'] = args.imputer
    if args.scaler:
        cfg['scaler'] = args.scaler
    if args.max_clusters:
        cfg['max_clusters'] = args.max_clusters
    if args.cluster_algo:
        cfg['cluster_algo'] = args.cluster_algo
    if args.schema:
        cfg['schema'] = args.schema

    # 1. Generate synthetic data
    print("Generating synthetic data...")
    # command-line or config may have set 'n_samples' and schema overrides
    n_samples = cfg.get('n_samples', 1000)
    schema = cfg.get('schema')
    if schema:
        # allow direct schema in config or path to a yaml
        if isinstance(schema, str) and os.path.isfile(schema):
            df = generate_from_yaml(schema)
        elif isinstance(schema, dict):
            df = generate_from_schema(schema)
        else:
            raise ValueError('Invalid schema in config')
    else:
        # Prefer a project-level sample_schema.yaml if present, but override 'n' with CLI/config n_samples
        sample_schema_path = 'sample_schema.yaml'
        if os.path.isfile(sample_schema_path):
            # load YAML, override n, and generate
            with open(sample_schema_path, 'r', encoding='utf-8') as sf:
                try:
                    sample_schema = yaml.safe_load(sf) or {}
                except Exception:
                    sample_schema = None
            if isinstance(sample_schema, dict):
                sample_schema['n'] = n_samples
                df = generate_from_schema(sample_schema)
            else:
                # fallback to inline default schema
                default_schema = {
                    'n': n_samples,
                    'fields': {
                        'timestamp': {'type': 'date', 'start': '2024-01-01', 'freq': 'H'},
                        'substation_id': {'type': 'categorical', 'categories': [f'SS{str(i).zfill(3)}' for i in range(1, 11)]},
                        'temperature': {'type': 'normal', 'mean': 25.0, 'std': 5.0},
                        'voltage': {'type': 'normal', 'mean': 230.0, 'std': 5.0},
                        'current': {'type': 'normal', 'mean': 10.0, 'std': 3.0},
                        'status': {'type': 'categorical', 'categories': ['OK', 'DEGRADED', 'FAULT'], 'weights': [0.85, 0.10, 0.05]}
                    }
                }
                df = generate_from_schema(default_schema)
        else:
            # no sample schema file: build a default power-grid schema and generate
            default_schema = {
                'n': n_samples,
                'fields': {
                    'timestamp': {'type': 'date', 'start': '2024-01-01', 'freq': 'H'},
                    'substation_id': {'type': 'categorical', 'categories': [f'SS{str(i).zfill(3)}' for i in range(1, 11)]},
                    'temperature': {'type': 'normal', 'mean': 25.0, 'std': 5.0},
                    'voltage': {'type': 'normal', 'mean': 230.0, 'std': 5.0},
                    'current': {'type': 'normal', 'mean': 10.0, 'std': 3.0},
                    'status': {'type': 'categorical', 'categories': ['OK', 'DEGRADED', 'FAULT'], 'weights': [0.85, 0.10, 0.05]}
                }
            }
            df = generate_from_schema(default_schema)
    raw_csv = 'raw_data.csv'
    df.to_csv(raw_csv, index=False)
    print(f"Raw data saved to {raw_csv}")
    print(f"Shape: {df.shape}")
    print("\nSample of raw data:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # 2. Ingest and preprocess
    print("\nIngesting and preprocessing...")
    di = DataIngestor()
    X_raw = di.ingest('csv', source=raw_csv)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # choose imputer/scaler from config or flags
    imputer_name = cfg.get('imputer', args.imputer)
    scaler_name = cfg.get('scaler', args.scaler)
    imputer = get_imputer(imputer_name)
    scaler = get_scaler(scaler_name)

    # impute
    X_imputed = imputer.fit_transform(df[numeric_cols])
    if scaler is not None:
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = X_imputed
    X_clean = pd.DataFrame(X_scaled, columns=numeric_cols, index=df.index)
    print(f"Preprocessed shape: {X_clean.shape}")

    # Optionally run hyperparameter optimization
    if args.hyperopt or cfg.get('hyperopt'):
        iters = cfg.get('hyperopt_iter', args.hyperopt_iter)
        print(f"Running hyperparameter optimization for {iters} iterations...")
        try:
            best_config, hist_df = hyperparam_optimization(raw_csv, iterations=int(iters), max_k=int(cfg.get('max_clusters', args.max_clusters)))
            print('Hyperopt best config:', best_config)
        except Exception as e:
            print('Hyperparameter optimization failed:', e)

    # 3. Dimension reduction
    print("\nApplying PCA...")
    X_reduced, pca = reduce_dimensions(X_clean, variance_threshold=0.95)
    print(f"Reduced dimensions shape: {X_reduced.shape}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    
    # 4. Find optimal clusters
    print("\nFinding optimal number of clusters...")
    max_k = cfg.get('max_clusters', args.max_clusters)
    optimal_k = find_optimal_clusters(X_reduced, max_clusters=int(max_k))
    print(f"Optimal number of clusters: {optimal_k}")
    
    # 5. Final clustering
    print("\nPerforming final clustering...")
    cluster_algo = cfg.get('cluster_algo', args.cluster_algo).lower()
    if cluster_algo == 'kmeans':
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_reduced)
    elif cluster_algo == 'dbscan' and DBSCAN is not None:
        db = DBSCAN()
        cluster_labels = db.fit_predict(X_reduced)
    elif cluster_algo == 'agglomerative' and AgglomerativeClustering is not None:
        agg = AgglomerativeClustering(n_clusters=optimal_k)
        cluster_labels = agg.fit_predict(X_reduced)
    else:
        print(f"Unknown or unavailable clustering algorithm '{cluster_algo}', falling back to kmeans")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_reduced)
    
    # 6. Engineer features and save enhanced dataset
    print("\nEngineering features...")
    enhanced_df = engineer_features(X_reduced, cluster_labels, df)
    enhanced_csv = 'enhanced_data.csv'
    enhanced_df.to_csv(enhanced_csv, index=False)
    print(f"\nEnhanced data saved to {enhanced_csv}")
    print(f"Final shape: {enhanced_df.shape}")
    print("\nNew features added:")
    print([col for col in enhanced_df.columns if col not in df.columns])

if __name__ == '__main__':
    main()