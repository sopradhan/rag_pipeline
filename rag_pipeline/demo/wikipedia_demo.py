"""Demo script showcasing RAG pipeline with Wikipedia data."""

import os
import sys
from typing import List
import wikipedia
from tqdm import tqdm
import streamlit as st

from rag_pipeline.ingestion.core import DataIngester, ProcessedChunk
from rag_pipeline.classify.hybrid import HybridClassifier
from rag_pipeline.embed.manager import EmbeddingManager
from rag_pipeline.security.manager import SecurityManager
from rag_pipeline.security.guardrails import ContentGuardrails
from rag_pipeline.monitoring.metrics import MetricsCollector
from rag_pipeline.monitoring.audit import AuditLogger, AuditEventType

class WikipediaDemo:
    """Demo class for RAG pipeline using Wikipedia data."""
    
    def __init__(
        self,
        topics: List[str] = None,
        model_config: Optional[Dict] = None
    ):
        self.topics = topics or [
            "Artificial Intelligence",
            "Machine Learning",
            "Natural Language Processing",
            "Computer Vision",
            "Deep Learning",
            "Neural Networks",
            "Robotics",
            "Data Science",
            "Big Data",
            "Cloud Computing"
        ]
        
        # Initialize model provider system
        from rag_pipeline.models.provider import ModelConfig, ModelProvider, ModelManager
        
        # Default model configuration with fallbacks
        default_config = {
            "provider_order": [
                ModelProvider.HUGGINGFACE,  # Try HuggingFace first
                ModelProvider.OPENAI,       # Fallback to OpenAI
                ModelProvider.GEMINI        # Finally try Gemini
            ],
            "model_configs": {
                ModelProvider.HUGGINGFACE: {
                    'llm_model': 'google/flan-t5-large',
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'use_sentence_transformers': True
                },
                ModelProvider.OPENAI: {
                    'llm_model': 'gpt-3.5-turbo-instruct',
                    'chat_model': 'gpt-3.5-turbo',
                    'embedding_model': 'text-embedding-ada-002',
                    'temperature': 0.3
                },
                ModelProvider.GEMINI: {
                    'llm_model': 'models/gemini-pro',
                    'chat_model': 'models/gemini-pro'
                }
            }
        }
        
        # Use provided config or default
        config = ModelConfig(**(model_config or default_config))
        self.model_manager = ModelManager(config)
        
        # Initialize components with model manager
        self.security = SecurityManager()
        self.guardrails = ContentGuardrails()
        self.classifier = HybridClassifier(
            rules_path="classify/rules.json",
            model_manager=self.model_manager  # Pass model manager instead of direct API key
        )
        self.embedding_manager = EmbeddingManager(model_manager=self.model_manager)
        self.metrics = MetricsCollector()
        self.audit = AuditLogger()
        
        # Create demo user
        self.demo_user = self.security.create_user(
            username="demo_user",
            roles=["reader", "analyst"]
        )
    
    def fetch_wikipedia_content(self) -> List[dict]:
        """Fetch content from Wikipedia for demo topics."""
        articles = []
        
        for topic in tqdm(self.topics, desc="Fetching Wikipedia articles"):
            try:
                # Get Wikipedia page
                page = wikipedia.page(topic)
                
                article = {
                    "title": page.title,
                    "content": page.content,
                    "url": page.url,
                    "references": page.references
                }
                
                articles.append(article)
                
                # Log success
                self.audit.log_event(
                    event_type=AuditEventType.DOCUMENT_INGEST,
                    user_id="demo_user",
                    description=f"Successfully fetched Wikipedia article: {topic}",
                    severity="LOW"
                )
                
            except Exception as e:
                print(f"Error fetching {topic}: {e}")
                self.audit.log_event(
                    event_type=AuditEventType.ERROR,
                    user_id="demo_user",
                    description=f"Failed to fetch Wikipedia article: {topic}",
                    severity="MEDIUM",
                    metadata={"error": str(e)}
                )
        
        return articles
    
    def process_articles(self, articles: List[dict]) -> List[ProcessedChunk]:
        """Process fetched articles through the pipeline."""
        chunks = []
        
        for article in tqdm(articles, desc="Processing articles"):
            try:
                # 1. Create chunks
                article_chunks = DataIngester.chunk_text(
                    text=article["content"],
                    chunk_size=500,
                    overlap=50
                )
                
                # 2. Process each chunk
                for i, chunk in enumerate(article_chunks):
                    # Apply content guardrails
                    filtered_text = self.guardrails.filter_content(chunk)
                    
                    # Create processed chunk
                    processed_chunk = ProcessedChunk(
                        content=filtered_text,
                        metadata={
                            "title": article["title"],
                            "url": article["url"],
                            "chunk_index": i,
                            "source": "wikipedia"
                        }
                    )
                    
                    # Classify content
                    classification = self.classifier.classify(filtered_text)
                    processed_chunk.metadata["classification"] = classification.model_dump()
                    
                    chunks.append(processed_chunk)
                    
                    # Record metrics
                    self.metrics.record_metric("chunks_processed", 1)
                    
            except Exception as e:
                print(f"Error processing {article['title']}: {e}")
                self.audit.log_event(
                    event_type=AuditEventType.ERROR,
                    user_id="demo_user",
                    description=f"Failed to process article: {article['title']}",
                    severity="MEDIUM",
                    metadata={"error": str(e)}
                )
        
        return chunks
    
    def create_embeddings_and_clusters(self, chunks: List[ProcessedChunk]):
        """Create embeddings and clusters from processed chunks."""
        try:
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings(
                [chunk.content for chunk in chunks]
            )
            
            # Store in vector database
            self.embedding_manager.store_embeddings(
                embeddings=embeddings,
                metadata=[chunk.metadata for chunk in chunks]
            )
            
            # Create clusters
            clusters = self.embedding_manager.create_clusters(embeddings)
            
            # Record metrics
            self.metrics.record_metric("embeddings_created", len(embeddings))
            self.metrics.record_metric("clusters_created", len(clusters))
            
            return clusters
            
        except Exception as e:
            print(f"Error in embedding/clustering: {e}")
            self.audit.log_event(
                event_type=AuditEventType.ERROR,
                user_id="demo_user",
                description="Failed to create embeddings/clusters",
                severity="HIGH",
                metadata={"error": str(e)}
            )
            return None
    
    def run_demo(self):
        """Run the complete demo pipeline."""
        st.title("RAG Pipeline Demo - Wikipedia Articles")
        
        # 1. Fetch articles
        st.header("1. Fetching Wikipedia Articles")
        articles = self.fetch_wikipedia_content()
        st.success(f"✅ Fetched {len(articles)} articles")
        
        # Show article titles
        st.subheader("Fetched Articles")
        for article in articles:
            st.write(f"- {article['title']}")
        
        # 2. Process articles
        st.header("2. Processing Articles")
        chunks = self.process_articles(articles)
        st.success(f"✅ Created {len(chunks)} chunks")
        
        # Show classification distribution
        st.subheader("Content Classification")
        classifications = {}
        for chunk in chunks:
            domain = chunk.metadata["classification"]["domain"]
            classifications[domain] = classifications.get(domain, 0) + 1
        
        st.bar_chart(classifications)
        
        # 3. Create embeddings and clusters
        st.header("3. Creating Embeddings and Clusters")
        clusters = self.create_embeddings_and_clusters(chunks)
        if clusters:
            st.success(f"✅ Created {len(clusters)} clusters")
            
            # Show cluster visualization
            st.subheader("Document Clusters")
            fig = self.embedding_manager.visualize_clusters(clusters)
            st.plotly_chart(fig)
        
        # 4. Show metrics
        st.header("4. System Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Documents",
                len(articles)
            )
        
        with col2:
            st.metric(
                "Total Chunks",
                len(chunks)
            )
        
        with col3:
            st.metric(
                "Total Clusters",
                len(clusters) if clusters else 0
            )
        
        # 5. Show sample query
        st.header("5. Try a Query")
        query = st.text_input("Enter your question:")
        
        if query:
            try:
                # Get relevant chunks
                results = self.embedding_manager.search(
                    query,
                    num_results=3
                )
                
                st.subheader("Relevant Results")
                for i, result in enumerate(results, 1):
                    st.markdown(f"**Result {i}**")
                    st.write(f"Source: {result.metadata['title']}")
                    st.write(f"Content: {result.content[:200]}...")
                    st.write(f"Domain: {result.metadata['classification']['domain']}")
                    st.write("---")
                
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    # Get API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Run demo
    demo = WikipediaDemo(openai_api_key=openai_api_key)
    demo.run_demo()