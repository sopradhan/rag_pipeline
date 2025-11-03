"""Core ingestion system for the RAG pipeline.

This module provides a flexible data ingestion system that can handle:
- Structured data (databases, CSV)
- Semi-structured data (JSON, XML, logs)
- Unstructured data (text, PDFs, docs)

Key features:
- Plugin architecture for data source adapters
- Automatic format detection
- Parallel processing for large datasets
- Content extraction and normalization
- Metadata enrichment
"""

from __future__ import annotations

import abc
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Set, Type, Union

import magic
import pandas as pd
from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

class ContentType(str, Enum):
    """Supported content types for ingestion."""
    TEXT = "text"
    PDF = "pdf"
    WORD = "docx"
    EXCEL = "xlsx"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    SQL = "sql"
    HTML = "html"
    MARKDOWN = "md"
    UNKNOWN = "unknown"

class DataSource(BaseModel):
    """Configuration for a data source."""
    name: str
    type: ContentType
    location: str
    credentials: Optional[Dict[str, str]] = None
    options: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedChunk:
    """A processed chunk of content with metadata."""
    content: str
    metadata: Dict[str, Any]
    source_type: ContentType
    chunk_id: str
    raw_data: Optional[bytes] = None

class IngestorPlugin(Protocol):
    """Protocol for data source adapter plugins."""
    
    def can_handle(self, source: DataSource) -> bool:
        """Check if this plugin can handle the data source."""
        ...
    
    def ingest(self, source: DataSource) -> Iterator[ProcessedChunk]:
        """Ingest data from the source."""
        ...

class BaseIngestor(abc.ABC):
    """Base class for data ingestors."""
    
    def __init__(self):
        self.supported_types: Set[ContentType] = set()
    
    @abc.abstractmethod
    def ingest(self, source: DataSource) -> Iterator[ProcessedChunk]:
        """Ingest data from the source."""
        pass

class SQLIngestor(BaseIngestor):
    """Ingest data from SQL databases."""
    
    def __init__(self):
        super().__init__()
        self.supported_types = {ContentType.SQL}
        self._engines: Dict[str, Engine] = {}
    
    def _get_engine(self, connection_string: str) -> Engine:
        """Get or create SQLAlchemy engine."""
        if connection_string not in self._engines:
            self._engines[connection_string] = create_engine(connection_string)
        return self._engines[connection_string]
    
    def ingest(self, source: DataSource) -> Iterator[ProcessedChunk]:
        if not source.credentials or "connection_string" not in source.credentials:
            raise ValueError("SQL source requires connection_string in credentials")
        
        engine = self._get_engine(source.credentials["connection_string"])
        query = source.location  # location field contains the SQL query
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            
            for row in result:
                # Convert row to dict
                data = dict(zip(columns, row))
                
                yield ProcessedChunk(
                    content=str(data),
                    metadata={
                        "source": source.name,
                        "type": "sql",
                        "columns": columns,
                    },
                    source_type=ContentType.SQL,
                    chunk_id=f"{source.name}_{id(row)}",
                    raw_data=None
                )

class FileIngestor(BaseIngestor):
    """Ingest data from files using LangChain loaders."""
    
    LOADER_MAP = {
        ContentType.PDF: PyPDFLoader,
        ContentType.WORD: UnstructuredWordDocumentLoader,
        ContentType.EXCEL: UnstructuredExcelLoader,
        ContentType.CSV: CSVLoader,
        ContentType.TEXT: TextLoader,
    }
    
    def __init__(self):
        super().__init__()
        self.supported_types = set(self.LOADER_MAP.keys())
    
    def _get_loader_cls(self, content_type: ContentType) -> Type[BaseLoader]:
        """Get the appropriate LangChain loader class."""
        if content_type not in self.LOADER_MAP:
            raise ValueError(f"Unsupported content type: {content_type}")
        return self.LOADER_MAP[content_type]
    
    def ingest(self, source: DataSource) -> Iterator[ProcessedChunk]:
        loader_cls = self._get_loader_cls(source.type)
        loader = loader_cls(source.location)
        
        for doc in loader.load():
            yield ProcessedChunk(
                content=doc.page_content,
                metadata={
                    "source": source.name,
                    "type": source.type,
                    **doc.metadata
                },
                source_type=source.type,
                chunk_id=f"{source.name}_{id(doc)}",
                raw_data=None
            )

class DirectoryIngestor(BaseIngestor):
    """Recursively ingest files from a directory."""
    
    def __init__(self):
        super().__init__()
        self.supported_types = {ContentType.TEXT}  # Can handle any text-based file
        self.file_ingestor = FileIngestor()
    
    def _detect_type(self, path: str) -> ContentType:
        """Detect content type using python-magic."""
        mime = magic.from_file(path, mime=True)
        
        # Map MIME types to ContentType
        mime_map = {
            "application/pdf": ContentType.PDF,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentType.WORD,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ContentType.EXCEL,
            "text/csv": ContentType.CSV,
            "text/plain": ContentType.TEXT,
            "application/json": ContentType.JSON,
            "text/xml": ContentType.XML,
            "text/html": ContentType.HTML,
            "text/markdown": ContentType.MARKDOWN,
        }
        
        return mime_map.get(mime, ContentType.UNKNOWN)
    
    def ingest(self, source: DataSource) -> Iterator[ProcessedChunk]:
        path = Path(source.location)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {source.location}")
        
        # Walk directory and process files
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    content_type = self._detect_type(str(file_path))
                    if content_type != ContentType.UNKNOWN:
                        # Create a new source for this file
                        file_source = DataSource(
                            name=f"{source.name}_{file_path.name}",
                            type=content_type,
                            location=str(file_path)
                        )
                        # Use FileIngestor to process the file
                        yield from self.file_ingestor.ingest(file_source)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

class DataIngestManager:
    """Orchestrates data ingestion from multiple sources."""
    
    def __init__(self):
        self.ingestors: Dict[ContentType, List[BaseIngestor]] = {}
        self.plugins: List[IngestorPlugin] = []
        
        # Register built-in ingestors
        self.register_ingestor(SQLIngestor())
        self.register_ingestor(FileIngestor())
        self.register_ingestor(DirectoryIngestor())
    
    def register_ingestor(self, ingestor: BaseIngestor):
        """Register an ingestor for specific content types."""
        for content_type in ingestor.supported_types:
            if content_type not in self.ingestors:
                self.ingestors[content_type] = []
            self.ingestors[content_type].append(ingestor)
    
    def register_plugin(self, plugin: IngestorPlugin):
        """Register a plugin ingestor."""
        self.plugins.append(plugin)
    
    def _get_ingestor(self, source: DataSource) -> Optional[BaseIngestor]:
        """Find an appropriate ingestor for the source."""
        # Check plugins first
        for plugin in self.plugins:
            if plugin.can_handle(source):
                return plugin
        
        # Then check built-in ingestors
        if source.type in self.ingestors:
            return self.ingestors[source.type][0]  # Use first registered ingestor
        
        return None
    
    def process_source(self, source: DataSource) -> Iterator[ProcessedChunk]:
        """Process a single data source."""
        ingestor = self._get_ingestor(source)
        if not ingestor:
            raise ValueError(f"No ingestor found for source type: {source.type}")
        
        yield from ingestor.ingest(source)
    
    def process_sources(self, sources: List[DataSource], max_workers: int = 4) -> Iterator[ProcessedChunk]:
        """Process multiple data sources in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(lambda s: list(self.process_source(s)), source)
                for source in sources
            ]
            
            for future in futures:
                try:
                    chunks = future.result()
                    yield from chunks
                except Exception as e:
                    logger.error(f"Error processing source: {e}")

def load_plugin(plugin_path: str) -> IngestorPlugin:
    """Dynamically load a plugin from a Python path string."""
    try:
        module_path, class_name = plugin_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)
        return plugin_class()
    except Exception as e:
        raise ImportError(f"Could not load plugin {plugin_path}: {e}")