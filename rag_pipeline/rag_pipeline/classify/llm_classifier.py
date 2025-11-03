"""LLM-based contextual classifier for enhanced text classification."""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from .hybrid import ContentDomain, Classification

class LLMClassifierConfig(BaseModel):
    """Configuration for LLM classifier."""
    
    temperature: float = 0.3
    model_name: str = "gpt-4"  # or gpt-3.5-turbo for faster, cheaper classification
    confidence_threshold: float = 0.7
    max_context_length: int = 1000
    
    # Classification prompts
    domain_prompt: str = """Analyze the following text and classify it into one of these domains:
    - TECHNICAL: Technical or engineering content
    - BUSINESS: Business-related content
    - LEGAL: Legal or compliance content
    - GENERAL: General information
    
    Text: {text}
    
    Provide your classification as a JSON object with:
    - domain: The domain name
    - confidence: A float between 0 and 1
    - reasoning: Brief explanation of your choice"""
    
    category_prompt: str = """For a {domain} text, classify it into the most appropriate category.
    
    Text: {text}
    
    Available categories for {domain}:
    {categories}
    
    Provide your classification as a JSON object with:
    - category: The category name
    - confidence: A float between 0 and 1
    - subcategories: List of relevant subcategories
    - tags: List of relevant tags
    - reasoning: Brief explanation of your choice"""
    
    # Category definitions by domain
    category_definitions: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "TECHNICAL": [
                "code", "architecture", "infrastructure", "data",
                "security", "api", "testing", "deployment", 
                "documentation", "other"
            ],
            "BUSINESS": [
                "sales", "marketing", "finance", "hr",
                "operations", "strategy", "product", "support", "other"
            ],
            "LEGAL": [
                "contracts", "compliance", "privacy", "licensing",
                "intellectual_property", "regulatory", "other"
            ],
            "GENERAL": [
                "news", "announcement", "policy", "process",
                "information", "other"
            ]
        }
    )

class LLMClassifier:
    """LLM-based classifier using advanced language models."""
    
    def __init__(
        self,
        config: Optional[LLMClassifierConfig] = None,
        api_key: Optional[str] = None
    ):
        self.config = config or LLMClassifierConfig()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=self.config.temperature,
            model_name=self.config.model_name,
            openai_api_key=api_key
        )
        
        # Create classification chains
        self.domain_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=self.config.domain_prompt,
                input_variables=["text"]
            )
        )
        
        self.category_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=self.config.category_prompt,
                input_variables=["domain", "text", "categories"]
            )
        )
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max context length while preserving meaning."""
        if len(text) <= self.config.max_context_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:self.config.max_context_length]
        last_period = truncated.rfind('.')
        
        if last_period > 0:
            return truncated[:last_period + 1]
        return truncated
    
    def classify_domain(self, text: str) -> Tuple[ContentDomain, float, str]:
        """Classify text domain using LLM."""
        truncated_text = self._truncate_text(text)
        
        try:
            result = self.domain_chain.run(text=truncated_text)
            parsed = json.loads(result)
            
            return (
                ContentDomain(parsed["domain"].lower()),
                parsed["confidence"],
                parsed["reasoning"]
            )
        except Exception as e:
            print(f"Domain classification error: {e}")
            return ContentDomain.UNKNOWN, 0.0, str(e)
    
    def classify_category(
        self,
        text: str,
        domain: ContentDomain
    ) -> Tuple[str, float, List[str], List[str], str]:
        """Classify text category within a domain using LLM."""
        truncated_text = self._truncate_text(text)
        categories = self.config.category_definitions[domain.upper()]
        
        try:
            result = self.category_chain.run(
                domain=domain.upper(),
                text=truncated_text,
                categories="\n".join(f"- {c}" for c in categories)
            )
            parsed = json.loads(result)
            
            return (
                parsed["category"],
                parsed["confidence"],
                parsed.get("subcategories", []),
                parsed.get("tags", []),
                parsed["reasoning"]
            )
        except Exception as e:
            print(f"Category classification error: {e}")
            return "unknown", 0.0, [], [], str(e)
    
    def classify(self, text: str) -> Classification:
        """Perform full LLM-based classification."""
        # Classify domain
        domain, domain_conf, domain_reason = self.classify_domain(text)
        
        # If confident about domain, classify category
        if domain_conf >= self.config.confidence_threshold:
            category, cat_conf, subcats, tags, cat_reason = (
                self.classify_category(text, domain)
            )
            
            # Use minimum confidence
            confidence = min(domain_conf, cat_conf)
            
            # Create metadata with reasoning
            metadata = {
                "domain_reasoning": domain_reason,
                "category_reasoning": cat_reason,
                "llm_model": self.config.model_name
            }
            
            return Classification(
                domain=domain,
                category=category,
                confidence=confidence,
                subcategories=subcats,
                tags=tags,
                metadata=metadata
            )
        
        # Fallback for low confidence
        return Classification(
            domain=ContentDomain.UNKNOWN,
            category="unknown",
            confidence=domain_conf,
            metadata={"domain_reasoning": domain_reason}
        )
    
    def batch_classify(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[Classification]:
        """Classify multiple texts efficiently in batches."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.classify(text) for text in batch]
            results.extend(batch_results)
        
        return results
        
# Example usage
if __name__ == "__main__":
    config = LLMClassifierConfig(
        temperature=0.3,
        model_name="gpt-3.5-turbo",  # or gpt-4 for higher accuracy
        confidence_threshold=0.7
    )
    
    classifier = LLMClassifier(config=config)
    
    # Example texts
    texts = [
        """The REST API implementation uses OAuth2 for authentication
        and includes rate limiting middleware for security.""",
        
        """Q4 revenue increased by 25% YoY, driven by strong enterprise
        sales and improved customer retention metrics.""",
        
        """Updated privacy policy to comply with GDPR requirements
        regarding data processing and user consent."""
    ]
    
    for text in texts:
        result = classifier.classify(text)
        print(f"\nText: {text.strip()}")
        print(f"Classification: {result.model_dump_json(indent=2)}")