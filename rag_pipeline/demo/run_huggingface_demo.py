"""Run RAG pipeline demo with Hugging Face models."""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline.config import Config
from rag_pipeline.models.provider import ModelConfig, ModelProvider
from rag_pipeline.demo.wikipedia_demo import WikipediaDemo

def setup_huggingface_demo():
    """Set up and run the demo with Hugging Face models."""
    
    # Load configuration
    config = Config()
    
    # Configure model preferences
    model_config = {
        "provider_order": [
            ModelProvider.HUGGINGFACE  # Only use Hugging Face
        ],
        "model_configs": {
            ModelProvider.HUGGINGFACE: {
                'llm_model': 'google/flan-t5-large',  # General language model
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',  # Efficient embedding model
                'use_sentence_transformers': True,
                'chat_model': 'facebook/blenderbot-400M-distill',  # For interactive responses
                'api_token': os.getenv('HUGGINGFACE_API_TOKEN')
            }
        }
    }
    
    # Set up topics for demo
    topics = [
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Neural Networks"
    ]
    
    # Create and run demo
    st.title("RAG Pipeline Demo - Hugging Face Models")
    
    # Show configuration
    st.sidebar.header("Model Configuration")
    st.sidebar.markdown("""
    **Using Hugging Face Models:**
    - LLM: `google/flan-t5-large`
    - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
    - Chat: `facebook/blenderbot-400M-distill`
    """)
    
    # Initialize demo
    try:
        demo = WikipediaDemo(
            topics=topics,
            model_config=model_config
        )
        
        # Run demo
        demo.run_demo()
        
    except Exception as e:
        st.error(f"Error initializing demo: {str(e)}")
        if st.button("Show Detailed Error"):
            st.code(str(e))
        
        # Check Hugging Face token
        if not os.getenv('HUGGINGFACE_API_TOKEN'):
            st.warning("""
            ⚠️ Hugging Face API token not found!
            
            Please set the HUGGINGFACE_API_TOKEN environment variable:
            1. Create a token at https://huggingface.co/settings/tokens
            2. Add it to your .env file:
               ```
               HUGGINGFACE_API_TOKEN=your_token_here
               ```
            """)

if __name__ == "__main__":
    # Set up environment
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    # Run demo
    setup_huggingface_demo()