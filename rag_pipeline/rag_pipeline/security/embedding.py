"""Integration of security with EmbeddingManager.

This module provides a secure wrapper around the EmbeddingManager that enforces
RBAC/ABAC permissions on all vector operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..embed.manager import DocumentChunk, EmbeddingManager
from .manager import Permission, SecurityManager, SecureVectorDB

class SecureEmbeddingManager:
    """Security wrapper for EmbeddingManager."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        security_manager: SecurityManager
    ):
        self.manager = embedding_manager
        self.security = security_manager
        self.secure_db = SecureVectorDB(security_manager)
    
    def add_documents(
        self,
        user_id: str,
        documents: List[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> bool:
        """Securely add documents to vector store."""
        # Check write permission
        if not self.secure_db.check_collection_access(
            user_id,
            collection_name or self.manager.db_config.collection_name,
            Permission.WRITE
        ):
            return False
        
        # Add documents
        self.manager.add_documents(documents)
        return True
    
    def search(
        self,
        user_id: str,
        query: str,
        n_results: int = 5,
        collection_name: Optional[str] = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict]:
        """Securely search for similar documents."""
        collection = collection_name or self.manager.db_config.collection_name
        
        # Check read permission
        if not self.secure_db.check_collection_access(
            user_id,
            collection,
            Permission.READ
        ):
            return []
        
        # Perform search
        results = self.manager.search(
            query,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        # Filter results based on user attributes
        return self.secure_db.filter_results(
            user_id,
            collection,
            results["documents"][0]
        )
    
    def delete_documents(
        self,
        user_id: str,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Securely delete documents from vector store."""
        # Check delete permission
        if not self.secure_db.check_collection_access(
            user_id,
            collection_name or self.manager.db_config.collection_name,
            Permission.DELETE
        ):
            return False
        
        # Delete documents
        self.manager.collection.delete(ids=ids)
        return True
    
    def get_user_collections(self, user_id: str) -> List[str]:
        """Get collections user has access to."""
        collections = []
        user = self.security.users.get(user_id)
        if not user:
            return collections
        
        # Check each collection
        for collection in self.manager.db.list_collections():
            if self.secure_db.check_collection_access(
                user_id,
                collection.name,
                Permission.READ
            ):
                collections.append(collection.name)
        
        return collections

# Example usage
if __name__ == "__main__":
    from ..embed.manager import EmbeddingConfig, EmbeddingManager
    
    # Initialize managers
    security = SecurityManager(jwt_secret="your-secret-key")
    embedding_manager = EmbeddingManager()
    secure_manager = SecureEmbeddingManager(embedding_manager, security)
    
    # Example documents
    docs = [
        DocumentChunk(
            text="Confidential sales report",
            metadata={
                "department": "Sales",
                "required_attributes": {"department": ["Sales"]}
            }
        ),
        DocumentChunk(
            text="Public company blog post",
            metadata={"public": True}
        )
    ]
    
    # Add test user
    from .manager import User, Role, Permission
    
    role = Role(
        name="sales",
        permissions={Permission.READ, Permission.WRITE},
        attributes={"department": ["Sales"]}
    )
    
    user = User(
        id="test_user",
        username="sales_user",
        roles=["sales"],
        attributes={"department": ["Sales"]}
    )
    
    security.add_role(role)
    security.add_user(user)
    
    # Test operations
    print("\nAdding documents...")
    success = secure_manager.add_documents("test_user", docs)
    print(f"Add success: {success}")
    
    print("\nSearching documents...")
    results = secure_manager.search("test_user", "sales", n_results=2)
    print(f"Search results: {results}")
    
    print("\nAccessible collections:")
    collections = secure_manager.get_user_collections("test_user")
    print(collections)