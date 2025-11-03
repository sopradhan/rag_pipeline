"""Integration of content guardrails with data ingestion and embedding pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..ingestion.core import DataSource, ProcessedChunk
from ..security.guardrails import Action, ContentGuardrails, ContentType, GuardrailsConfig

class SecureProcessor:
    """Secure content processor with guardrails."""
    
    def __init__(
        self,
        guardrails: Optional[ContentGuardrails] = None,
        config: Optional[GuardrailsConfig] = None
    ):
        self.guardrails = guardrails or ContentGuardrails(config=config)
    
    def process_chunk(
        self,
        chunk: ProcessedChunk,
        content_types: Optional[Set[ContentType]] = None
    ) -> Tuple[ProcessedChunk, bool]:
        """Process a chunk with content guardrails.
        
        Returns:
            Tuple of (processed_chunk, is_blocked)
        """
        # Apply content filtering
        filtered_text, results = self.guardrails.filter_content(
            chunk.content,
            content_types=content_types,
            document_id=chunk.chunk_id
        )
        
        # Check if content should be blocked
        is_blocked = any(r.action == Action.BLOCK for r in results)
        if is_blocked:
            return chunk, True
        
        # Update chunk with filtered content
        chunk.content = filtered_text
        
        # Add detection results to metadata
        chunk.metadata["detection_results"] = [
            {
                "rule_id": r.rule_id,
                "content_type": r.content_type,
                "action": r.action,
                "confidence": r.confidence
            }
            for r in results
        ]
        
        return chunk, False
    
    def process_chunks(
        self,
        chunks: List[ProcessedChunk],
        content_types: Optional[Set[ContentType]] = None
    ) -> Tuple[List[ProcessedChunk], List[ProcessedChunk]]:
        """Process multiple chunks with content guardrails.
        
        Returns:
            Tuple of (allowed_chunks, blocked_chunks)
        """
        allowed = []
        blocked = []
        
        for chunk in chunks:
            processed, is_blocked = self.process_chunk(chunk, content_types)
            if is_blocked:
                blocked.append(processed)
            else:
                allowed.append(processed)
        
        return allowed, blocked

class SecureDataSource(DataSource):
    """Data source with content security checks."""
    
    class Config:
        arbitrary_types_allowed = True
    
    security_config: Optional[GuardrailsConfig] = None
    allowed_content_types: Set[ContentType] = set()
    processor: Optional[SecureProcessor] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.processor:
            self.processor = SecureProcessor(
                config=self.security_config
            )
    
    def filter_content(self, content: str) -> Tuple[str, bool]:
        """Filter content and check if it should be blocked."""
        filtered, results = self.processor.guardrails.filter_content(
            content,
            content_types=self.allowed_content_types
        )
        is_blocked = any(r.action == Action.BLOCK for r in results)
        return filtered, is_blocked

# Example usage
if __name__ == "__main__":
    # Create processor with custom config
    config = GuardrailsConfig(
        enabled_types={ContentType.PII, ContentType.SECURITY},
        default_action=Action.MASK,
        audit_logging=True
    )
    
    processor = SecureProcessor(config=config)
    
    # Example chunks
    chunks = [
        ProcessedChunk(
            content="Email: test@example.com\nAPI Key: secret_123",
            metadata={"source": "doc1"},
            source_type="text",
            chunk_id="chunk1"
        ),
        ProcessedChunk(
            content="Normal text without sensitive data",
            metadata={"source": "doc2"},
            source_type="text",
            chunk_id="chunk2"
        ),
        ProcessedChunk(
            content="OFFENSIVE CONTENT HERE",
            metadata={"source": "doc3"},
            source_type="text",
            chunk_id="chunk3"
        )
    ]
    
    # Process chunks
    allowed, blocked = processor.process_chunks(
        chunks,
        content_types={ContentType.PII, ContentType.SECURITY, ContentType.OFFENSIVE}
    )
    
    print("\nAllowed Chunks:")
    for chunk in allowed:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Content: {chunk.content}")
        print("Detection Results:", chunk.metadata.get("detection_results"))
    
    print("\nBlocked Chunks:")
    for chunk in blocked:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print("Original content blocked due to policy violation")