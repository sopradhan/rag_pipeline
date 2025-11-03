"""Embedding generation and vector storage system.

Features:
- Multi-modal embedding support (text, images)
- Efficient chunking and embedding generation
- Vector database integration (ChromaDB)
- Clustering and visualization
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import chromadb
import numpy as np
import pandas as pd
import torch
from chromadb.config import Settings
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..classify.hybrid import Classification

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    device: str = "cpu"  # or "cuda" if available

class ClusteringConfig(BaseModel):
    """Configuration for embedding clustering."""
    algorithm: str = "kmeans"
    n_clusters: Optional[int] = None
    min_clusters: int = 2
    max_clusters: int = 20
    random_state: int = 42

class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    collection_name: str = "documents"
    persist_directory: Optional[str] = None
    distance_function: str = "cosine"

class DocumentChunk(BaseModel):
    """A chunk of text with metadata and classification."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: Optional[List[float]] = None
    classification: Optional[Classification] = None
    metadata: Dict = Field(default_factory=dict)
    cluster_id: Optional[int] = None

class EmbeddingManager:
    """Manages embedding generation and vector storage."""
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
        clustering_config: Optional[ClusteringConfig] = None,
        db_config: Optional[VectorDBConfig] = None
    ):
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.clustering_config = clustering_config or ClusteringConfig()
        self.db_config = db_config or VectorDBConfig()
        
        # Initialize embedding model
        self.model = SentenceTransformer(
            self.embedding_config.model_name,
            device=self.embedding_config.device
        )
        
        # Initialize vector DB
        self.db = chromadb.Client(Settings(
            persist_directory=self.db_config.persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.db.get_or_create_collection(
            name=self.db_config.collection_name,
            metadata={"hnsw:space": self.db_config.distance_function}
        )
        
        self.scaler = StandardScaler()
        self._current_clusters: Optional[MiniBatchKMeans] = None
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        batch_size = batch_size or self.embedding_config.batch_size
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.embedding_config.normalize
        )
        
        return embeddings
    
    def chunk_and_embed(
        self,
        documents: List[DocumentChunk],
        batch_size: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Chunk documents and generate embeddings."""
        # Get all texts
        texts = [doc.text for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size)
        
        # Update documents with embeddings
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb.tolist()
        
        return documents
    
    def _find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_k: int,
        max_k: int
    ) -> Tuple[int, float]:
        """Find optimal number of clusters using silhouette score."""
        best_k = min_k
        best_score = -1
        
        for k in range(min_k, max_k + 1):
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.clustering_config.random_state
            )
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k, best_score
    
    def cluster_embeddings(
        self,
        documents: List[DocumentChunk],
        n_clusters: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Cluster document embeddings."""
        if not documents:
            return documents
        
        # Get embeddings array
        embeddings = np.array([doc.embedding for doc in documents])
        
        # Scale embeddings
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters, _ = self._find_optimal_clusters(
                scaled_embeddings,
                self.clustering_config.min_clusters,
                self.clustering_config.max_clusters
            )
        
        # Perform clustering
        self._current_clusters = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.clustering_config.random_state
        )
        labels = self._current_clusters.fit_predict(scaled_embeddings)
        
        # Update documents with cluster IDs
        for doc, label in zip(documents, labels):
            doc.cluster_id = int(label)
        
        return documents
    
    def add_documents(
        self,
        documents: List[DocumentChunk],
        batch_size: Optional[int] = None
    ):
        """Add documents to vector store."""
        # Ensure documents have embeddings
        if any(doc.embedding is None for doc in documents):
            documents = self.chunk_and_embed(documents, batch_size)
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [
            {
                **doc.metadata,
                "cluster_id": str(doc.cluster_id) if doc.cluster_id is not None else None,
                "classification": doc.classification.model_dump() if doc.classification else None
            }
            for doc in documents
        ]
        texts = [doc.text for doc in documents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def get_cluster_stats(self) -> pd.DataFrame:
        """Get statistics about clusters."""
        if not self._current_clusters:
            return pd.DataFrame()
        
        # Get all documents and their cluster IDs
        results = self.collection.get()
        cluster_ids = [
            int(meta.get("cluster_id", -1))
            for meta in results["metadatas"]
        ]
        
        # Create DataFrame
        df = pd.DataFrame({
            "cluster_id": cluster_ids,
            "document_id": results["ids"],
            "text": results["documents"]
        })
        
        # Compute stats
        stats = df.groupby("cluster_id").agg({
            "document_id": "count"
        }).rename(columns={"document_id": "size"})
        
        # Add centroids
        centroids = pd.DataFrame(
            self._current_clusters.cluster_centers_,
            index=range(self._current_clusters.n_clusters)
        )
        stats = stats.join(centroids.add_prefix("centroid_"))
        
        return stats
    
    def save_state(self, path: Union[str, Path]):
        """Save current state (embeddings, clusters) to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save ChromaDB
        if self.db_config.persist_directory:
            self.db.persist()
        
        # Save clustering state if available
        if self._current_clusters:
            import joblib
            cluster_path = path / "clustering_state.joblib"
            joblib.dump(
                {
                    "clusters": self._current_clusters,
                    "scaler": self.scaler
                },
                cluster_path
            )
    
    def load_state(self, path: Union[str, Path]):
        """Load saved state from disk."""
        path = Path(path)
        
        # ChromaDB auto-loads if persist_directory is set
        
        # Load clustering state if available
        cluster_path = path / "clustering_state.joblib"
        if cluster_path.exists():
            import joblib
            state = joblib.load(cluster_path)
            self._current_clusters = state["clusters"]
            self.scaler = state["scaler"]

# Example usage
if __name__ == "__main__":
    # Create manager
    manager = EmbeddingManager(
        embedding_config=EmbeddingConfig(device="cuda" if torch.cuda.is_available() else "cpu"),
        db_config=VectorDBConfig(persist_directory="./vector_store")
    )
    
    # Example documents
    docs = [
        DocumentChunk(
            text="This is a technical document about APIs",
            metadata={"source": "tech_docs"}
        ),
        DocumentChunk(
            text="Sales report for Q4 2023",
            metadata={"source": "reports"}
        )
    ]
    
    # Process documents
    docs = manager.chunk_and_embed(docs)
    docs = manager.cluster_embeddings(docs)
    manager.add_documents(docs)
    
    # Search example
    results = manager.search("api documentation", n_results=2)
    print("\nSearch results:")
    for doc in results["documents"][0]:
        print(f"- {doc}")