from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure Chroma persistence to E:/epoch_explorer/chroma
# Try to create a persistent Chroma client. Newer Chroma releases changed configuration
# and may raise a ValueError if an old on-disk layout exists. We catch that and fall
# back to an in-memory client so the demo still runs.
try:
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="E:/epoch_explorer/chroma"))
except ValueError as e:
    print("Chroma persistent client failed:", e)
    print("Falling back to an in-memory Chroma client for the demo.")
    print("If you want persistent storage, run: pip install chroma-migrate && chroma-migrate")
    client = chromadb.Client()

model = SentenceTransformer("all-MiniLM-L6-v2")

# sample documents
docs = [
    "This report describes an Overheat fault and steps: Check coolant, reduce load, inspect bearings.",
    "Normal operation with small temperature variance and stable power draw.",
    "Power surge observed; recommended to check input transformers and fuses."
]
ids = ["doc1", "doc2", "doc3"]

# compute embeddings
embs = model.encode(docs, convert_to_numpy=True)

# create or get collection
collection = client.get_or_create_collection(name="smartdairy_reports")

# add documents
collection.add(documents=docs, ids=ids, embeddings=embs.tolist())

# query
query_text = "Overheat corrective measures"
q_emb = model.encode([query_text], convert_to_numpy=True)

results = collection.query(query_embeddings=q_emb.tolist(), n_results=2)
print("Results:", results)
