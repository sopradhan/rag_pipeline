"""
Benchmark script for Chroma: compares cosine (via normalized embeddings) vs Euclidean (raw embeddings)
- Generates synthetic embeddings (random vectors) to simulate document vectors
- Indexes them in Chroma (in-memory)
- Runs a set of query vectors and measures recall@k against brute-force ground truth (numpy-based)
- Measures indexing time and query latency

Run with:
& .venv/Scripts/python.exe benchmark_chroma.py

"""

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import chromadb
from chromadb.config import Settings

# Settings
N_DOCS = 10000  # number of documents to index
DIM = 384       # embedding dimension (typical MiniLM)
N_QUERIES = 200
K = 10

# create in-memory chroma client
client = chromadb.Client()

# create collection names
col_cos = client.get_or_create_collection(name='bench_cosine')
col_euc = client.get_or_create_collection(name='bench_euclidean')

# Generate synthetic random vectors for reproducibility
rng = np.random.RandomState(42)
docs = [f'doc_{i}' for i in range(N_DOCS)]
embs = rng.randn(N_DOCS, DIM).astype(np.float32)

# Prepare normalized vectors for cosine experiments
norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

# Index in Chroma
print('Indexing documents (cosine-normalized) into bench_cosine...')
start = time.time()
col_cos.add(ids=docs, documents=[""]*N_DOCS, embeddings=norm_embs.tolist())
print('Done in', time.time()-start, 's')

print('Indexing documents (raw) into bench_euclidean...')
start = time.time()
col_euc.add(ids=docs, documents=[""]*N_DOCS, embeddings=embs.tolist())
print('Done in', time.time()-start, 's')

# Create query set (some near duplicates + random)
queries = rng.randn(N_QUERIES, DIM).astype(np.float32)
# normalize queries for cosine
q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)

# Ground truth (brute force)
print('Computing brute-force ground truth (numpy)...')
start = time.time()
# cosine similarity ground truth (higher is better)
cos_sim = cosine_similarity(q_norm, norm_embs)
cos_true_topk = np.argsort(-cos_sim, axis=1)[:, :K]
# euclidean ground truth (lower is better)
euc_dist = euclidean_distances(queries, embs)
euc_true_topk = np.argsort(euc_dist, axis=1)[:, :K]
print('Ground truth computed in', time.time()-start, 's')

# Query Chroma and measure recall@K
print('Querying Chroma and measuring recall@K...')
recalls_cos = []
latencies_cos = []
recalls_euc = []
latencies_euc = []

for i in range(N_QUERIES):
    # cosine: query with normalized vector
    t0 = time.time()
    res = col_cos.query(query_embeddings=[q_norm[i].tolist()], n_results=K)
    t1 = time.time()
    latencies_cos.append(t1-t0)
    returned_ids = res['ids'][0]
    true_ids = [docs[idx] for idx in cos_true_topk[i]]
    # recall@K
    recall = len(set(returned_ids).intersection(set(true_ids))) / K
    recalls_cos.append(recall)

    # euclidean
    t0 = time.time()
    res2 = col_euc.query(query_embeddings=[queries[i].tolist()], n_results=K)
    t1 = time.time()
    latencies_euc.append(t1-t0)
    returned_ids2 = res2['ids'][0]
    true_ids2 = [docs[idx] for idx in euc_true_topk[i]]
    recall2 = len(set(returned_ids2).intersection(set(true_ids2))) / K
    recalls_euc.append(recall2)

print('\nBenchmark results (N_DOCS={}, DIM={}, N_QUERIES={}, K={})'.format(N_DOCS, DIM, N_QUERIES, K))
print('Cosine: avg recall@{} = {:.4f}, avg latency = {:.4f}s'.format(K, np.mean(recalls_cos), np.mean(latencies_cos)))
print('Euclidean: avg recall@{} = {:.4f}, avg latency = {:.4f}s'.format(K, np.mean(recalls_euc), np.mean(latencies_euc)))

# Save a small report
report = {
    'N_DOCS': N_DOCS,
    'DIM': DIM,
    'N_QUERIES': N_QUERIES,
    'K': K,
    'cosine': {'recall': float(np.mean(recalls_cos)), 'latency': float(np.mean(latencies_cos))},
    'euclidean': {'recall': float(np.mean(recalls_euc)), 'latency': float(np.mean(latencies_euc))}
}

import json
with open('chroma_benchmark_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Report saved to chroma_benchmark_report.json')
