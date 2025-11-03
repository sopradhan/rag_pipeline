"""Configuration management for the RAG pipeline."""

from typing import Any, Dict, List, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings, Field

class SecurityConfig(BaseSettings):
    """Security-related configuration."""
    jwt_secret_key: str = Field(..., env='JWT_SECRET_KEY')
    jwt_algorithm: str = Field('HS256', env='JWT_ALGORITHM')
    jwt_expiration_hours: int = Field(24, env='JWT_EXPIRATION_HOURS')

class OpenAIConfig(BaseSettings):
    """OpenAI-specific configuration."""
    api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    model_name: str = Field('gpt-4', env='OPENAI_MODEL_NAME')
    embedding_model: str = Field('text-embedding-ada-002', env='OPENAI_EMBEDDING_MODEL')
    temperature: float = Field(0.3, env='OPENAI_TEMPERATURE')

class HuggingFaceConfig(BaseSettings):
    """HuggingFace-specific configuration."""
    api_token: Optional[str] = Field(None, env='HUGGINGFACE_API_TOKEN')
    llm_model: str = Field('google/flan-t5-large', env='HUGGINGFACE_LLM_MODEL')
    embedding_model: str = Field(
        'sentence-transformers/all-MiniLM-L6-v2',
        env='HUGGINGFACE_EMBEDDING_MODEL'
    )

class GeminiConfig(BaseSettings):
    """Google/Gemini-specific configuration."""
    api_key: Optional[str] = Field(None, env='GOOGLE_API_KEY')
    model_name: str = Field('models/gemini-pro', env='GEMINI_MODEL_NAME')

class AzureOpenAIConfig(BaseSettings):
    """Azure OpenAI-specific configuration."""
    api_key: Optional[str] = Field(None, env='AZURE_OPENAI_API_KEY')
    endpoint: Optional[str] = Field(None, env='AZURE_OPENAI_ENDPOINT')
    llm_deployment: str = Field(..., env='AZURE_OPENAI_LLM_DEPLOYMENT')
    chat_deployment: str = Field(..., env='AZURE_OPENAI_CHAT_DEPLOYMENT')
    embedding_deployment: str = Field(..., env='AZURE_OPENAI_EMBEDDING_DEPLOYMENT')

class VectorDBConfig(BaseSettings):
    """Vector database configuration."""
    db_path: Path = Field('./data/chromadb', env='CHROMA_DB_PATH')
    host: str = Field('localhost', env='CHROMA_DB_HOST')
    port: int = Field(8000, env='CHROMA_DB_PORT')

class GuardrailsConfig(BaseSettings):
    """Content guardrails configuration."""
    pii_detection_threshold: float = Field(0.8, env='PII_DETECTION_THRESHOLD')
    content_filter_threshold: float = Field(0.7, env='CONTENT_FILTER_THRESHOLD')
    enable_content_filtering: bool = Field(True, env='ENABLE_CONTENT_FILTERING')

class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration."""
    enable_metrics: bool = Field(True, env='ENABLE_METRICS')
    metrics_retention_days: int = Field(30, env='METRICS_RETENTION_DAYS')
    log_level: str = Field('INFO', env='LOG_LEVEL')

class DemoConfig(BaseSettings):
    """Demo-specific configuration."""
    wikipedia_max_articles: int = Field(10, env='WIKIPEDIA_MAX_ARTICLES')
    chunk_size: int = Field(500, env='CHUNK_SIZE')
    chunk_overlap: int = Field(50, env='CHUNK_OVERLAP')

class SystemConfig(BaseSettings):
    """System-wide configuration."""
    cache_dir: Path = Field('./cache', env='CACHE_DIR')
    max_retries: int = Field(3, env='MAX_RETRIES')
    timeout_seconds: int = Field(30, env='TIMEOUT_SECONDS')
    enable_debug: bool = Field(False, env='ENABLE_DEBUG')

class AppConfig(BaseSettings):
    """Application-level configuration."""
    name: str = Field('RAG Pipeline', env='APP_NAME')
    version: str = Field('1.0.0', env='APP_VERSION')
    environment: str = Field('development', env='APP_ENVIRONMENT')

class Config:
    """Main configuration class."""
    
    def __init__(self, env_file: str = '.env'):
        # Load environment variables
        load_dotenv(env_file)
        
        # Initialize all configurations
        self.security = SecurityConfig()
        self.openai = OpenAIConfig()
        self.huggingface = HuggingFaceConfig()
        self.gemini = GeminiConfig()
        self.azure_openai = AzureOpenAIConfig()
        self.vector_db = VectorDBConfig()
        self.guardrails = GuardrailsConfig()
        self.monitoring = MonitoringConfig()
        self.demo = DemoConfig()
        self.system = SystemConfig()
        self.app = AppConfig()
        
        # Parse provider order
        self.model_provider_order = self._parse_provider_order()
    
    def _parse_provider_order(self) -> List[str]:
        """Parse the provider order from environment variable."""
        order = os.getenv('MODEL_PROVIDER_ORDER', 'huggingface,openai,gemini')
        return [provider.strip() for provider in order.split(',')]
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if provider == 'openai':
            return self.openai.dict()
        elif provider == 'huggingface':
            return self.huggingface.dict()
        elif provider == 'gemini':
            return self.gemini.dict()
        elif provider == 'azure_openai':
            return self.azure_openai.dict()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get complete model configuration."""
        return {
            "provider_order": self.model_provider_order,
            "model_configs": {
                provider: self.get_provider_config(provider)
                for provider in self.model_provider_order
            }
        }
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.environment == 'production'
    
    @property
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.system.enable_debug

# Global configuration instance
config = Config()

if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print("=====================")
    print(f"Environment: {config.app.environment}")
    print(f"Model Provider Order: {config.model_provider_order}")
    print("\nOpenAI Configuration:")
    print(f"Model: {config.openai.model_name}")
    print(f"Embedding Model: {config.openai.embedding_model}")
    print("\nHuggingFace Configuration:")
    print(f"Model: {config.huggingface.llm_model}")
    print(f"Embedding Model: {config.huggingface.embedding_model}")
    print("\nSystem Configuration:")
    print(f"Cache Directory: {config.system.cache_dir}")
    print(f"Debug Enabled: {config.system.enable_debug}")
    print(f"Max Retries: {config.system.max_retries}")