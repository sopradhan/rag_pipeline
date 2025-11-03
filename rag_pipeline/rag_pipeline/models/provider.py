"""Model provider management for LLMs and embeddings with fallback support."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import os
from abc import ABC, abstractmethod

import openai
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure_openai"

class ModelType(str, Enum):
    """Types of models."""
    LLM = "llm"
    CHAT = "chat"
    EMBEDDING = "embedding"

class ModelConfig:
    """Configuration for model providers."""
    
    def __init__(
        self,
        provider_order: List[ModelProvider],
        model_configs: Dict[ModelProvider, Dict[str, Any]]
    ):
        self.provider_order = provider_order
        self.model_configs = model_configs
        
        # Initialize API keys and configurations
        self._init_providers()
    
    def _init_providers(self):
        """Initialize all configured providers."""
        for provider in self.provider_order:
            config = self.model_configs.get(provider, {})
            
            if provider == ModelProvider.OPENAI:
                openai.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
            
            elif provider == ModelProvider.HUGGINGFACE:
                os.environ['HUGGINGFACE_API_TOKEN'] = (
                    config.get('api_key') or os.getenv('HUGGINGFACE_API_TOKEN')
                )
            
            elif provider == ModelProvider.GEMINI:
                genai.configure(api_key=config.get('api_key') or os.getenv('GOOGLE_API_KEY'))
            
            elif provider == ModelProvider.AZURE_OPENAI:
                os.environ['AZURE_OPENAI_API_KEY'] = (
                    config.get('api_key') or os.getenv('AZURE_OPENAI_API_KEY')
                )
                os.environ['AZURE_OPENAI_ENDPOINT'] = (
                    config.get('endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
                )

class BaseModelProvider(ABC):
    """Base class for model providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def get_llm(self) -> Any:
        """Get LLM model."""
        pass
    
    @abstractmethod
    def get_chat_model(self) -> Any:
        """Get chat model."""
        pass
    
    @abstractmethod
    def get_embedding_model(self) -> Any:
        """Get embedding model."""
        pass

class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider."""
    
    def get_llm(self) -> OpenAI:
        return OpenAI(
            model_name=self.config.get('llm_model', 'gpt-3.5-turbo-instruct'),
            temperature=self.config.get('temperature', 0.3)
        )
    
    def get_chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.config.get('chat_model', 'gpt-3.5-turbo'),
            temperature=self.config.get('temperature', 0.3)
        )
    
    def get_embedding_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.config.get('embedding_model', 'text-embedding-ada-002')
        )

class HuggingFaceProvider(BaseModelProvider):
    """HuggingFace model provider."""
    
    def get_llm(self) -> HuggingFaceHub:
        return HuggingFaceHub(
            repo_id=self.config.get('llm_model', 'google/flan-t5-large'),
            task="text-generation"
        )
    
    def get_chat_model(self) -> HuggingFaceHub:
        return HuggingFaceHub(
            repo_id=self.config.get('chat_model', 'google/flan-t5-large'),
            task="text2text-generation"
        )
    
    def get_embedding_model(self) -> Union[HuggingFaceEmbeddings, SentenceTransformer]:
        model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        if self.config.get('use_sentence_transformers', True):
            return SentenceTransformer(model_name)
        return HuggingFaceEmbeddings(model_name=model_name)

class GeminiProvider(BaseModelProvider):
    """Google's Gemini model provider."""
    
    def get_llm(self) -> Any:
        model_name = self.config.get('llm_model', 'models/gemini-pro')
        return genai.GenerativeModel(model_name)
    
    def get_chat_model(self) -> Any:
        model_name = self.config.get('chat_model', 'models/gemini-pro')
        return genai.GenerativeModel(model_name)
    
    def get_embedding_model(self) -> Any:
        # Gemini currently doesn't provide embeddings
        # Fallback to another provider's embedding model
        raise NotImplementedError("Gemini does not provide embedding models")

class AzureOpenAIProvider(BaseModelProvider):
    """Azure OpenAI model provider."""
    
    def get_llm(self) -> Any:
        return OpenAI(
            deployment_name=self.config.get('llm_deployment'),
            model_name=self.config.get('llm_model', 'gpt-35-turbo'),
            openai_api_type="azure"
        )
    
    def get_chat_model(self) -> Any:
        return ChatOpenAI(
            deployment_name=self.config.get('chat_deployment'),
            model_name=self.config.get('chat_model', 'gpt-35-turbo'),
            openai_api_type="azure"
        )
    
    def get_embedding_model(self) -> Any:
        return OpenAIEmbeddings(
            deployment=self.config.get('embedding_deployment'),
            model=self.config.get('embedding_model', 'text-embedding-ada-002'),
            openai_api_type="azure"
        )

class ModelManager:
    """Manager for handling model providers with fallback."""
    
    PROVIDER_CLASSES = {
        ModelProvider.OPENAI: OpenAIProvider,
        ModelProvider.HUGGINGFACE: HuggingFaceProvider,
        ModelProvider.GEMINI: GeminiProvider,
        ModelProvider.AZURE_OPENAI: AzureOpenAIProvider
    }
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.providers = {}
        
        # Initialize providers
        self._init_providers()
    
    def _init_providers(self):
        """Initialize all configured providers."""
        for provider in self.config.provider_order:
            provider_config = self.config.model_configs.get(provider, {})
            provider_class = self.PROVIDER_CLASSES.get(provider)
            
            if provider_class:
                try:
                    self.providers[provider] = provider_class(provider_config)
                except Exception as e:
                    print(f"Failed to initialize {provider}: {e}")
    
    def get_model(
        self,
        model_type: ModelType,
        provider: Optional[ModelProvider] = None
    ) -> Any:
        """Get a model with fallback support."""
        if provider:
            # Try specific provider first
            try:
                return self._get_model_from_provider(provider, model_type)
            except Exception as e:
                print(f"Failed to get model from {provider}: {e}")
        
        # Try providers in order
        for provider in self.config.provider_order:
            try:
                return self._get_model_from_provider(provider, model_type)
            except Exception as e:
                print(f"Failed to get model from {provider}: {e}")
                continue
        
        raise RuntimeError(f"No available provider for {model_type}")
    
    def _get_model_from_provider(
        self,
        provider: ModelProvider,
        model_type: ModelType
    ) -> Any:
        """Get a specific type of model from a provider."""
        provider_instance = self.providers.get(provider)
        if not provider_instance:
            raise ValueError(f"Provider {provider} not initialized")
        
        if model_type == ModelType.LLM:
            return provider_instance.get_llm()
        elif model_type == ModelType.CHAT:
            return provider_instance.get_chat_model()
        elif model_type == ModelType.EMBEDDING:
            return provider_instance.get_embedding_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Example usage
if __name__ == "__main__":
    # Configure providers
    config = ModelConfig(
        provider_order=[
            ModelProvider.HUGGINGFACE,  # Try HuggingFace first
            ModelProvider.OPENAI,       # Fallback to OpenAI
            ModelProvider.GEMINI        # Finally try Gemini
        ],
        model_configs={
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
    )
    
    # Create model manager
    manager = ModelManager(config)
    
    # Get models with fallback
    try:
        # Try to get an LLM
        llm = manager.get_model(ModelType.LLM)
        print(f"Got LLM: {type(llm)}")
        
        # Try to get an embedding model
        embedding_model = manager.get_model(ModelType.EMBEDDING)
        print(f"Got embedding model: {type(embedding_model)}")
        
        # Try specific provider
        openai_chat = manager.get_model(
            ModelType.CHAT,
            provider=ModelProvider.OPENAI
        )
        print(f"Got OpenAI chat model: {type(openai_chat)}")
        
    except Exception as e:
        print(f"Error: {e}")