"""Content guardrails system with PII detection and content filtering.

Features:
- Rule-based PII detection (regex patterns)
- ML-based sensitive content detection
- Content masking and filtering
- Policy-based blocking
- Audit logging
"""

from __future__ import annotations

import datetime
import re
import uuid
from enum import Enum
from typing import Dict, List, Optional, Pattern, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from transformers import pipeline

class ContentType(str, Enum):
    """Types of sensitive content."""
    PII = "pii"
    FINANCIAL = "financial"
    HEALTH = "health"
    SECURITY = "security"
    OFFENSIVE = "offensive"
    CUSTOM = "custom"

class Action(str, Enum):
    """Actions to take on detected content."""
    MASK = "mask"
    REMOVE = "remove"
    BLOCK = "block"
    LOG = "log"
    ALLOW = "allow"

class PIIType(str, Enum):
    """Types of PII data."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DOB = "dob"
    IP_ADDRESS = "ip_address"
    CUSTOM = "custom"

class DetectionRule(BaseModel):
    """Rule definition for content detection."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    content_type: ContentType
    pattern: str
    is_regex: bool = True
    case_sensitive: bool = False
    action: Action
    mask_char: str = "*"
    description: Optional[str] = None
    
    _compiled_pattern: Optional[Pattern] = None
    
    def compile(self):
        """Compile regex pattern."""
        if self.is_regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._compiled_pattern = re.compile(self.pattern, flags)
    
    def match(self, text: str) -> List[Tuple[int, int, str]]:
        """Find all matches in text. Returns list of (start, end, matched_text)."""
        if self.is_regex:
            if not self._compiled_pattern:
                self.compile()
            return [(m.start(), m.end(), m.group()) for m in self._compiled_pattern.finditer(text)]
        else:
            # Simple substring search
            matches = []
            start = 0
            pattern = self.pattern if self.case_sensitive else self.pattern.lower()
            text_to_search = text if self.case_sensitive else text.lower()
            
            while True:
                idx = text_to_search.find(pattern, start)
                if idx == -1:
                    break
                matches.append((idx, idx + len(pattern), text[idx:idx + len(pattern)]))
                start = idx + 1
            
            return matches

class DetectionResult(BaseModel):
    """Result of content detection."""
    rule_id: str
    content_type: ContentType
    matches: List[Tuple[int, int, str]]
    action: Action
    confidence: float = 1.0

class AuditLog(BaseModel):
    """Audit log entry for content detection."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    content_type: ContentType
    action: Action
    rule_id: str
    num_matches: int
    sample_match: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)

class GuardrailsConfig(BaseModel):
    """Configuration for content guardrails."""
    enabled_types: Set[ContentType] = Field(default_factory=lambda: set(ContentType))
    default_action: Action = Action.MASK
    mask_char: str = "*"
    min_confidence: float = 0.8
    audit_logging: bool = True
    ml_detection: bool = True

class ContentGuardrails:
    """Content detection and filtering system."""
    
    # Common regex patterns for PII
    DEFAULT_PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        PIIType.SSN: r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        PIIType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def __init__(
        self,
        config: Optional[GuardrailsConfig] = None,
        rules: Optional[List[DetectionRule]] = None
    ):
        self.config = config or GuardrailsConfig()
        self.rules: Dict[str, DetectionRule] = {}
        self.audit_logs: List[AuditLog] = []
        
        # Add default PII rules
        self._add_default_rules()
        
        # Add custom rules
        if rules:
            for rule in rules:
                self.add_rule(rule)
        
        # Initialize ML detection if enabled
        self.ml_detector = None
        if self.config.ml_detection:
            try:
                self.ml_detector = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    return_all_scores=True
                )
            except Exception as e:
                print(f"Warning: Could not load ML detector: {e}")
    
    def _add_default_rules(self):
        """Add default PII detection rules."""
        for pii_type, pattern in self.DEFAULT_PATTERNS.items():
            rule = DetectionRule(
                name=f"default_{pii_type}",
                content_type=ContentType.PII,
                pattern=pattern,
                action=self.config.default_action,
                description=f"Default {pii_type} detector"
            )
            self.add_rule(rule)
    
    def add_rule(self, rule: DetectionRule):
        """Add a detection rule."""
        if rule.is_regex:
            rule.compile()
        self.rules[rule.id] = rule
    
    def _log_detection(
        self,
        rule_id: str,
        content_type: ContentType,
        action: Action,
        matches: List[Tuple[int, int, str]],
        document_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log a detection event."""
        if not self.config.audit_logging:
            return
        
        log = AuditLog(
            content_type=content_type,
            action=action,
            rule_id=rule_id,
            num_matches=len(matches),
            sample_match=matches[0][2] if matches else None,
            document_id=document_id,
            metadata=metadata or {}
        )
        self.audit_logs.append(log)
    
    def _apply_ml_detection(self, text: str) -> List[DetectionResult]:
        """Apply ML-based content detection."""
        if not self.ml_detector:
            return []
        
        try:
            # Get toxicity scores
            results = self.ml_detector(text)
            
            # Convert to DetectionResult format
            detected = []
            for scores in results[0]:
                label = scores["label"]
                confidence = scores["score"]
                
                if confidence >= self.config.min_confidence:
                    # Map toxic labels to content types
                    content_type = ContentType.OFFENSIVE
                    
                    result = DetectionResult(
                        rule_id="ml_toxic_" + label,
                        content_type=content_type,
                        matches=[(0, len(text), text)],  # Mark whole text
                        action=Action.BLOCK if confidence > 0.9 else Action.LOG,
                        confidence=confidence
                    )
                    detected.append(result)
            
            return detected
        except Exception as e:
            print(f"ML detection error: {e}")
            return []
    
    def detect(
        self,
        text: str,
        content_types: Optional[Set[ContentType]] = None,
        document_id: Optional[str] = None
    ) -> List[DetectionResult]:
        """Detect sensitive content in text."""
        if not text:
            return []
        
        content_types = content_types or self.config.enabled_types
        results = []
        
        # Apply rule-based detection
        for rule in self.rules.values():
            if rule.content_type in content_types:
                matches = rule.match(text)
                if matches:
                    result = DetectionResult(
                        rule_id=rule.id,
                        content_type=rule.content_type,
                        matches=matches,
                        action=rule.action
                    )
                    results.append(result)
                    
                    # Log detection
                    self._log_detection(
                        rule.id,
                        rule.content_type,
                        rule.action,
                        matches,
                        document_id
                    )
        
        # Apply ML detection if enabled
        if self.config.ml_detection and ContentType.OFFENSIVE in content_types:
            ml_results = self._apply_ml_detection(text)
            results.extend(ml_results)
            
            # Log ML detections
            for res in ml_results:
                self._log_detection(
                    res.rule_id,
                    res.content_type,
                    res.action,
                    res.matches,
                    document_id
                )
        
        return results
    
    def mask_content(
        self,
        text: str,
        results: List[DetectionResult]
    ) -> str:
        """Mask detected content in text."""
        if not results:
            return text
        
        # Sort matches by start position (reversed to process from end)
        all_matches = []
        for result in results:
            if result.action == Action.MASK:
                all_matches.extend(result.matches)
        
        all_matches.sort(key=lambda x: x[0], reverse=True)
        
        # Apply masking
        masked = list(text)
        for start, end, _ in all_matches:
            masked[start:end] = self.config.mask_char * (end - start)
        
        return ''.join(masked)
    
    def filter_content(
        self,
        text: str,
        content_types: Optional[Set[ContentType]] = None,
        document_id: Optional[str] = None
    ) -> Tuple[str, List[DetectionResult]]:
        """Detect and filter sensitive content."""
        # Detect sensitive content
        results = self.detect(text, content_types, document_id)
        
        # Check for blocking
        for result in results:
            if result.action == Action.BLOCK:
                return '', results
        
        # Apply masking
        filtered_text = self.mask_content(text, results)
        
        return filtered_text, results

# Example usage
if __name__ == "__main__":
    # Create guardrails with default config
    guardrails = ContentGuardrails()
    
    # Add custom rule
    custom_rule = DetectionRule(
        name="custom_token",
        content_type=ContentType.SECURITY,
        pattern=r'\b(api_key|token|secret)=\S+\b',
        action=Action.MASK,
        description="Detect API keys and tokens"
    )
    guardrails.add_rule(custom_rule)
    
    # Test detection
    test_text = """
    Contact: john.doe@example.com
    Phone: (123) 456-7890
    SSN: 123-45-6789
    API Key: api_key=1234567890abcdef
    Credit Card: 4111-1111-1111-1111
    
    Some offensive content here that should be detected by ML.
    """
    
    filtered, results = guardrails.filter_content(
        test_text,
        document_id="test_doc"
    )
    
    print("\nDetection Results:")
    for result in results:
        print(f"\nRule: {result.rule_id}")
        print(f"Type: {result.content_type}")
        print(f"Action: {result.action}")
        print("Matches:", [m[2] for m in result.matches])
    
    print("\nFiltered Text:")
    print(filtered)
    
    print("\nAudit Logs:")
    for log in guardrails.audit_logs:
        print(f"\n{log.timestamp}: {log.content_type} ({log.num_matches} matches)")