from typing import Dict, List, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os

class VectorDBClient:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.collection_name = "incidents"
        self.initialize_collection()

    def initialize_collection(self):
        """Initialize the vector collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Create new collection
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # Embedding dimension
                    distance=Distance.COSINE
                )
            )

    async def store(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Store a vector with its metadata.
        Returns the ID of the stored vector.
        """
        point_id = np.random.randint(0, 2**63)  # Generate random ID
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata
        )
        
        # Upload point
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        return str(point_id)

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional filtering.
        Returns list of matches with their metadata and scores.
        """
        # Convert filter conditions to Qdrant format if provided
        search_conditions = None
        if filter_conditions:
            search_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )
        
        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=search_conditions
        )
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                'id': str(hit.id),
                'score': hit.score,
                'metadata': hit.payload
            })
        
        return results

    async def batch_store(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple vectors with their metadata in batch.
        Returns list of stored vector IDs.
        """
        points = []
        ids = []
        
        for embedding, metadata in zip(embeddings, metadata_list):
            point_id = np.random.randint(0, 2**63)
            ids.append(str(point_id))
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            ))
        
        # Batch upload
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids

    async def delete(self, vector_ids: List[str]):
        """Delete vectors by their IDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[int(vid) for vid in vector_ids]
            )
        )

    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]):
        """Update metadata for a specific vector."""
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[int(vector_id)]
        )