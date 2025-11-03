"""Classification system for content categorization.

This module provides a hybrid classification system that combines:
- Rule-based classification using regex and keyword matching
- ML-based classification using embeddings and trained models
- Domain-specific categorization (technical, business, etc.)
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ContentDomain(str, Enum):
    """High-level content domains."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    LEGAL = "legal"
    GENERAL = "general"
    UNKNOWN = "unknown"

class TechnicalCategory(str, Enum):
    """Technical content categories."""
    CODE = "code"
    ARCHITECTURE = "architecture"
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    SECURITY = "security"
    API = "api"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    OTHER = "other"

class BusinessCategory(str, Enum):
    """Business content categories."""
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    HR = "hr"
    OPERATIONS = "operations"
    STRATEGY = "strategy"
    PRODUCT = "product"
    SUPPORT = "support"
    OTHER = "other"

class Classification(BaseModel):
    """Complete classification result."""
    domain: ContentDomain
    category: str
    confidence: float
    subcategories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, any] = Field(default_factory=dict)

class RuleDefinition(BaseModel):
    """Definition of a classification rule."""
    pattern: str
    domain: ContentDomain
    category: str
    subcategories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    priority: int = 1
    is_regex: bool = False

class HybridClassifier:
    """Combines rule-based, ML-based, and LLM-based classification."""
    
    def __init__(
        self,
        rules_path: Optional[Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
        ml_threshold: float = 0.6,
        llm_config: Optional[Dict] = None,
        openai_api_key: Optional[str] = None
    ):
        self.rules: List[RuleDefinition] = []
        self.ml_threshold = ml_threshold
        
        # Initialize LLM classifier
        try:
            from .llm_classifier import LLMClassifier, LLMClassifierConfig
            llm_config = llm_config or {}
            self.llm_classifier = LLMClassifier(
                config=LLMClassifierConfig(**llm_config),
                api_key=openai_api_key
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM classifier: {e}")
            self.llm_classifier = None
        
        # Load rules if provided
        if rules_path:
            self.load_rules(rules_path)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize ML classifier (trained separately)
        self.ml_classifier: Optional[BaseEstimator] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Cache compiled regex patterns
        self._regex_cache: Dict[str, re.Pattern] = {}
    
    def load_rules(self, rules_path: Path):
        """Load classification rules from JSON file."""
        with open(rules_path) as f:
            rules_data = json.load(f)
            self.rules = [RuleDefinition(**rule) for rule in rules_data]
        
        # Sort rules by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def _compile_regex(self, pattern: str) -> re.Pattern:
        """Get or compile regex pattern."""
        if pattern not in self._regex_cache:
            self._regex_cache[pattern] = re.compile(pattern, re.IGNORECASE)
        return self._regex_cache[pattern]
    
    def _apply_rules(self, text: str) -> Optional[Classification]:
        """Apply rule-based classification."""
        for rule in self.rules:
            if rule.is_regex:
                pattern = self._compile_regex(rule.pattern)
                if pattern.search(text):
                    return Classification(
                        domain=rule.domain,
                        category=rule.category,
                        confidence=1.0,
                        subcategories=rule.subcategories,
                        tags=rule.tags
                    )
            else:
                # Simple keyword matching
                if rule.pattern.lower() in text.lower():
                    return Classification(
                        domain=rule.domain,
                        category=rule.category,
                        confidence=1.0,
                        subcategories=rule.subcategories,
                        tags=rule.tags
                    )
        return None
    
    def _ml_classify(self, text: str) -> Optional[Classification]:
        """Apply ML-based classification."""
        if not self.embedding_model or not self.ml_classifier:
            return None
        
        try:
            # Get text embedding
            embedding = self.embedding_model.encode(text, show_progress_bar=False)
            
            # Get predictions and probabilities
            pred_idx = self.ml_classifier.predict([embedding])[0]
            probs = self.ml_classifier.predict_proba([embedding])[0]
            confidence = probs[pred_idx]
            
            if confidence >= self.ml_threshold:
                category = self.label_encoder.inverse_transform([pred_idx])[0]
                # Determine domain based on category
                domain = (
                    ContentDomain.TECHNICAL
                    if category in TechnicalCategory.__members__
                    else ContentDomain.BUSINESS
                    if category in BusinessCategory.__members__
                    else ContentDomain.GENERAL
                )
                
                return Classification(
                    domain=domain,
                    category=category,
                    confidence=float(confidence)
                )
        except Exception as e:
            print(f"ML classification error: {e}")
        return None
    
    def train_ml_classifier(
        self,
        texts: List[str],
        labels: List[str],
        model: Optional[BaseEstimator] = None
    ):
        """Train the ML classifier component."""
        if not self.embedding_model:
            raise ValueError("Embedding model not available")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        # Get embeddings for all texts
        X = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Initialize and train classifier
        self.ml_classifier = model or RandomForestClassifier(n_estimators=100)
        self.ml_classifier.fit(X, y)
    
    def _llm_classify(self, text: str) -> Optional[Classification]:
        """Apply LLM-based classification."""
        if not self.llm_classifier:
            return None
            
        try:
            return self.llm_classifier.classify(text)
        except Exception as e:
            print(f"LLM classification error: {e}")
            return None

    def classify(self, text: str) -> Classification:
        """Classify text using all available methods and Crew AI for final decision."""
        classifications = []
        
        # Get rule-based classification
        rule_result = self._apply_rules(text)
        if rule_result:
            rule_result.metadata["method"] = "rule-based"
            classifications.append(rule_result)
        
        # Get ML-based classification
        ml_result = self._ml_classify(text)
        if ml_result:
            ml_result.metadata["method"] = "ml-based"
            classifications.append(ml_result)
        
        # Get LLM-based classification
        llm_result = self._llm_classify(text)
        if llm_result:
            llm_result.metadata["method"] = "llm-based"
            classifications.append(llm_result)
        
        # Use Crew AI for final decision if we have multiple classifications
        if len(classifications) > 1:
            try:
                from .crew_classifier import CrewClassifier
                
                # Initialize Crew AI with same API key as LLM classifier
                crew = CrewClassifier(
                    openai_api_key=self.llm_classifier.config.api_key if self.llm_classifier else None,
                    model_name=self.llm_classifier.config.model_name if self.llm_classifier else "gpt-4",
                    temperature=0.3
                )
                
                # Get final classification from Crew AI
                final_result = crew.classify(text)
                
                # Add previous classifications to metadata
                final_result.metadata["previous_classifications"] = [
                    {
                        "domain": c.domain,
                        "category": c.category,
                        "confidence": c.confidence,
                        "method": c.metadata.get("method", "unknown")
                    }
                    for c in classifications
                ]
                
                return final_result
                
            except Exception as e:
                print(f"Crew AI error: {e}")
                # Fall back to highest confidence classification
                return max(classifications, key=lambda x: x.confidence)
        
        # If we only have one classification, use it
        elif len(classifications) == 1:
            return classifications[0]
        
        # Final fallback
        return Classification(
            domain=ContentDomain.UNKNOWN,
            category="unknown",
            confidence=0.0,
            metadata={"method": "fallback"}
        )
    
    def batch_classify(self, texts: List[str]) -> List[Classification]:
        """Classify multiple texts efficiently."""
        return [self.classify(text) for text in texts]

# Example usage
if __name__ == "__main__":
    # Example rules
    example_rules = [
        {
            "pattern": r"\b(api|rest|graphql|endpoint)\b",
            "domain": "technical",
            "category": "api",
            "is_regex": True,
            "priority": 2
        },
        {
            "pattern": r"\b(sales|revenue|customer)\b",
            "domain": "business",
            "category": "sales",
            "is_regex": True,
            "priority": 1
        }
    ]
    
    # Save example rules
    rules_file = Path("example_rules.json")
    with open(rules_file, "w") as f:
        json.dump(example_rules, f, indent=2)
    
    # Create classifier
    classifier = HybridClassifier(rules_path=rules_file)
    
    # Example classifications
    texts = [
        "The REST API endpoint needs authentication",
        "Q4 sales revenue exceeded targets",
        "General information about the company"
    ]
    
    for text in texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Classification: {result.model_dump_json(indent=2)}")