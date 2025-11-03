Chroma DB — Guide, Operations & Best Practices
===============================================

This document explains how to use Chroma as a local/embedded vector store, how embeddings are produced, where files are stored, how to configure persistence and cache locations, and recommended best practices for production and development.

Sections
- What is Chroma?
- Install & quick start
- Client creation (persistent vs in-memory)
- Embeddings: local vs remote
- Where models and caches live (move cache off C:)
- Choosing a distance/metric and normalization
- Operational best practices (batching, persistence, backups)
- Migration & compatibility (chroma-migrate)
- Example code (sentence-transformers + Chroma, OpenAI embeddings + Chroma)

What is Chroma?
---------------
Chroma is a lightweight vector database designed for in-process/local usage. It stores vectors alongside documents and metadata and provides nearest-neighbor search capabilities. Chroma is ideal for development, experimentation, and small-to-medium production workloads where running an external vector DB server is unnecessary.

Install & quick start
---------------------
Install the Python packages (run in your project's `.venv`):

```powershell
& .venv/Scripts/python.exe -m pip install chromadb sentence-transformers
```

Client creation — persistent vs in-memory
----------------------------------------
- Persistent (recommended for repeatable demos and small production): provide a `persist_directory` on a fast disk (preferably not C:).
- In-memory (good for quick tests): create a default client without `persist_directory`.

Example (persistent preferred):

```python
import chromadb
from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="E:/epoch_explorer/chroma"  # change to your drive
    )
)
```

If Chroma complains about deprecated on-disk layout, see the Migration section below.

Embeddings: local vs remote
---------------------------
1. Local embeddings (download once, run locally):
   - Libraries: `sentence-transformers`, `transformers`.
   - Pros: offline operation after first download, low per-call cost, control over model.
   - Cons: model weights take disk space and may be large.

2. Remote/API-based embeddings (OpenAI, Cohere, etc.):
   - Pros: no local model management, fast to get started.
   - Cons: per-call cost, privacy considerations (data sent to provider).

Where models and cache live (move off C:)
-----------------------------------------
Hugging Face models are cached by default under `%USERPROFILE%\.cache\huggingface` on Windows. You can redirect HF cache to another drive using environment variables:

PowerShell (temporary for session):

```powershell
$env:HF_HOME="E:/epoch_explorer/.cache"
$env:TRANSFORMERS_CACHE="E:/epoch_explorer/.cache"
```

Or put the same entries into your project's `.env` and use `python-dotenv` (we added `.env` earlier):

```
HF_HOME=E:/epoch_explorer/.cache
TRANSFORMERS_CACHE=E:/epoch_explorer/.cache
```

Choosing distance/metric and normalization
-----------------------------------------
Two common similarity metrics are used for semantic search:

- Cosine similarity (recommended for semantic similarity):
  - Measures angle between vectors; scale-invariant.
  - Use L2-normalized vectors when performing cosine comparisons for stable results.
- Euclidean (L2) distance:
  - Measures absolute distance; works well when vectors are not normalized.

Practical guidance:
- If your embedding model returns un-normalized embeddings and you want cosine similarity, L2-normalize them before storing and querying.
  Example using NumPy:
  ```python
  import numpy as np
  from sklearn.preprocessing import normalize

  embs = model.encode(docs, convert_to_numpy=True)
  embs_norm = normalize(embs, norm='l2')  # for cosine
  ```
- Some vector DBs accept a `metric` or `distance` parameter when creating a collection; if Chroma exposes this through its API or adapter, set it to `cosine` or `euclidean` accordingly. If not, normalize manually for cosine.

Operational best practices
-------------------------
- Batch adds/updates: encode documents in batches to avoid repeated model overhead.
- Reuse embeddings: persist embeddings with documents to avoid re-embedding on every run.
- Persist data on a non-system drive: use `persist_directory` on E:/ or D:/ to save space on C:.
- Backups: periodically copy the `persist_directory` to a safe backup location.
- Migration: when upgrading Chroma, run `chroma-migrate` if asked by the library.
- Monitoring: watch disk usage and index file growth; compact or re-index periodically if necessary.

Migration & compatibility (chroma-migrate)
------------------------------------------
If you see a message that your Chroma on-disk layout is deprecated, Chroma provides a migration tool:

```powershell
& .venv/Scripts/python.exe -m pip install chroma-migrate
chroma-migrate
```

Follow the tool's instructions to migrate old data to the new layout and then instantiate the persistent client normally.

Example code — sentence-transformers (local embeddings)
------------------------------------------------------
```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="E:/epoch_explorer/chroma"))
model = SentenceTransformer("all-MiniLM-L6-v2")

docs = ["Overheat: check coolant", "All OK"]
ids = ["d1","d2"]
embs = model.encode(docs, convert_to_numpy=True)

# Normalize if you plan to use cosine
from sklearn.preprocessing import normalize
embs_norm = normalize(embs, norm='l2')

collection = client.get_or_create_collection(name="reports")
collection.add(documents=docs, ids=ids, embeddings=embs_norm.tolist())

q_emb = normalize(model.encode(["Overheat corrective measures"], convert_to_numpy=True), norm='l2')
print(collection.query(query_embeddings=q_emb.tolist(), n_results=3))
```

Example code — OpenAI embeddings (remote)
-----------------------------------------
```python
import openai
import chromadb
from chromadb.config import Settings

openai.api_key = "sk_xxx"
client = chromadb.Client(Settings(persist_directory="E:/epoch_explorer/chroma"))
collection = client.get_or_create_collection(name="reports")

# helper
def openai_embed(texts, model="text-embedding-3-small"):
    resp = openai.Embedding.create(input=texts, model=model)
    return [d["embedding"] for d in resp["data"]]

texts = ["Overheat: check coolant", "All OK"]
ids = ["d1","d2"]
embs = openai_embed(texts)
collection.add(documents=texts, ids=ids, embeddings=embs)

q_emb = openai_embed(["Overheat corrective measures"])
print(collection.query(query_embeddings=q_emb, n_results=3))
```

Adapter pattern for CrewAI / other tools
---------------------------------------
If a library expects a `vectorstore` object, implement a thin adapter that exposes `add` and `search` (or `query`) calling Chroma under the hood. Example skeleton:

```python
class ChromaAdapter:
    def __init__(self, collection, embedding_model=None):
        self.collection = collection
        self.embedding_model = embedding_model
    def add(self, doc_id, text, metadata=None):
        emb = self.embedding_model.encode([text], convert_to_numpy=True).tolist()
        self.collection.add(documents=[text], ids=[doc_id], embeddings=emb, metadatas=[metadata])
    def search(self, query, k=5):
        q_emb = self.embedding_model.encode([query], convert_to_numpy=True).tolist()
        return self.collection.query(query_embeddings=q_emb, n_results=k)
```

Security & privacy
------------------
- If using remote embeddings (OpenAI), be mindful that text is sent to the provider.
- For local models, keep the HF cache on secure drives and restrict access permissions.

Performance tuning
------------------
- Use a small, fast model for large-scale indexing (e.g. `all-MiniLM-L6-v2`) or a larger model for improved quality.
- Batch encoding and use efficient hardware (GPU) when available.
- If you need very large-scale, consider deploying a dedicated vector DB (Qdrant, Milvus) instead of local Chroma.

FAQ
---
Q: Do I need internet to run embeddings every time?
A: For local models, you only need internet for the initial download. For API embeddings, yes — every request.

Q: How do I choose cosine vs euclidean?
A: For semantic similarity, cosine is typically better. If you use cosine, L2-normalize embeddings before storing/querying.

Q: Where should I store Chroma files?
A: Use a dedicated directory on a non-system drive (E:/ or D:/). Set `persist_directory` accordingly.

Closing notes
-------------
Chroma is an excellent option for development and small deployments. For production-grade scaling, evaluate managed or server vector DBs. If you want, I can:
- Integrate Chroma into `nw_demo.py`, replacing the Qdrant adapter.
- Add a persistent `persist_directory` on `E:/epoch_explorer/chroma` and demonstrate migration with `chroma-migrate`.

Which of the above would you like me to do next? 
