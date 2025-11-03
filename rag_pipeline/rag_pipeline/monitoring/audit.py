"""Audit logging system for RAG pipeline."""

from datetime import datetime
from typing import Dict, List, Optional, Union
import json
from pathlib import Path

class AuditEvent:
    """Audit event record."""
    
    def __init__(
        self,
        event_type: str,
        user_id: str,
        description: str,
        severity: str = "LOW",
        metadata: Optional[Dict] = None
    ):
        self.timestamp = datetime.now().isoformat()
        self.event_type = event_type
        self.user_id = user_id
        self.description = description
        self.severity = severity
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "description": self.description,
            "severity": self.severity,
            "metadata": self.metadata
        }

class AuditLogger:
    """System for logging and querying audit events."""
    
    def __init__(self, log_file: str = "audit.json"):
        self.log_file = Path(log_file)
        self._ensure_log_file()
        self.events: List[Dict] = []
        self._load_events()
    
    def _ensure_log_file(self):
        """Create log file if it doesn't exist."""
        if not self.log_file.exists():
            self.log_file.write_text("[]")
    
    def _load_events(self):
        """Load events from log file."""
        try:
            self.events = json.loads(self.log_file.read_text())
        except json.JSONDecodeError:
            self.events = []
    
    def _save_events(self):
        """Save events to log file."""
        self.log_file.write_text(json.dumps(self.events, indent=2))
    
    def log_event(
        self,
        event_type: str,
        user_id: str,
        description: str,
        severity: str = "LOW",
        metadata: Optional[Dict] = None
    ):
        """Log a new audit event."""
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            description=description,
            severity=severity,
            metadata=metadata
        )
        
        self.events.append(event.to_dict())
        self._save_events()
    
    def get_logs(
        self,
        event_type: Optional[List[str]] = None,
        severity: Optional[List[str]] = None,
        user: Optional[List[str]] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Query audit logs with filters."""
        filtered = self.events
        
        # Filter by event type
        if event_type:
            filtered = [
                e for e in filtered
                if e["event_type"] in event_type
            ]
        
        # Filter by severity
        if severity:
            filtered = [
                e for e in filtered
                if e["severity"] in severity
            ]
        
        # Filter by user
        if user:
            filtered = [
                e for e in filtered
                if e["user_id"] in user
            ]
        
        # Filter by time range
        if start_time:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e["timestamp"]) >= start_time
            ]
        
        if end_time:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e["timestamp"]) <= end_time
            ]
        
        # Filter by search term
        if search:
            search = search.lower()
            filtered = [
                e for e in filtered
                if (
                    search in e["description"].lower()
                    or search in e["event_type"].lower()
                    or search in e["user_id"].lower()
                    or any(
                        search in str(v).lower()
                        for v in e["metadata"].values()
                    )
                )
            ]
        
        return filtered
    
    def get_event_types(self) -> List[str]:
        """Get list of unique event types."""
        return sorted(list({e["event_type"] for e in self.events}))
    
    def get_users(self) -> List[str]:
        """Get list of unique users."""
        return sorted(list({e["user_id"] for e in self.events}))
    
    def clear_old_logs(self, days: int = 365):
        """Clear logs older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        self.events = [
            e for e in self.events
            if datetime.fromisoformat(e["timestamp"]) >= cutoff
        ]
        
        self._save_events()

# Example audit event types
class AuditEventType:
    """Common audit event types."""
    
    # Security events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    PERMISSION_CHANGE = "PERMISSION_CHANGE"
    
    # Content events
    DOCUMENT_INGEST = "DOCUMENT_INGEST"
    DOCUMENT_DELETE = "DOCUMENT_DELETE"
    DOCUMENT_UPDATE = "DOCUMENT_UPDATE"
    
    # Classification events
    CLASSIFICATION_ADD = "CLASSIFICATION_ADD"
    CLASSIFICATION_UPDATE = "CLASSIFICATION_UPDATE"
    
    # Embedding events
    EMBEDDING_GENERATE = "EMBEDDING_GENERATE"
    CLUSTER_CREATE = "CLUSTER_CREATE"
    CLUSTER_UPDATE = "CLUSTER_UPDATE"
    
    # Security events
    PII_DETECTED = "PII_DETECTED"
    CONTENT_BLOCKED = "CONTENT_BLOCKED"
    GUARDRAIL_TRIGGER = "GUARDRAIL_TRIGGER"
    
    # System events
    CONFIG_CHANGE = "CONFIG_CHANGE"
    ERROR = "ERROR"
    WARNING = "WARNING"