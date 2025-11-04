from typing import Dict, Any, Optional
from datetime import datetime
import json
from sqlalchemy.orm import Session
from .models import IncidentLog, SystemMetric, AlertEmail, QueueTask, IncidentSeverity
from ..ai.classification import classify_severity
from ..ai.embedding import generate_embeddings
from ..vector_db.client import VectorDBClient

class DataIngester:
    def __init__(self, db_session: Session, vector_db: VectorDBClient):
        self.db = db_session
        self.vector_db = vector_db

    async def ingest_incident(self, incident_data: Dict[str, Any]) -> IncidentLog:
        """
        Ingest an incident, classify severity, generate embeddings, and store in both SQL and vector DB.
        """
        # Classify severity using BERT model
        severity = await classify_severity(incident_data['description'])
        
        # Generate embeddings for vector DB
        embeddings = await generate_embeddings(incident_data['description'])
        
        # Store in vector DB with metadata
        vector_id = await self.vector_db.store(
            embeddings,
            metadata={
                'incident_type': incident_data.get('type'),
                'severity': severity.value,
                'timestamp': datetime.utcnow().isoformat(),
                'department': incident_data.get('department'),
                'project': incident_data.get('project')
            }
        )
        
        # Create incident log in SQL DB
        incident_log = IncidentLog(
            incident_json=incident_data,
            severity=severity,
            metadata={
                'vector_id': vector_id,
                'department': incident_data.get('department'),
                'project': incident_data.get('project')
            }
        )
        
        self.db.add(incident_log)
        self.db.commit()
        
        # Create processing task in queue
        task = QueueTask(
            task_json={
                'action': 'process_incident',
                'incident_id': incident_log.id,
                'vector_id': vector_id
            },
            status='pending',
            agent_type='triage',
            priority=self._calculate_priority(severity)
        )
        
        self.db.add(task)
        self.db.commit()
        
        return incident_log

    async def ingest_metric(self, metric_data: Dict[str, Any]) -> SystemMetric:
        """
        Ingest system metrics and store in SQL DB.
        """
        metric = SystemMetric(
            metrics_json=metric_data,
            metadata={
                'system': metric_data.get('system'),
                'metric_type': metric_data.get('type')
            }
        )
        
        self.db.add(metric)
        self.db.commit()
        return metric

    async def ingest_alert(self, alert_data: Dict[str, Any]) -> AlertEmail:
        """
        Ingest alert emails and store in SQL DB.
        """
        alert = AlertEmail(
            email_payload_json=alert_data,
            metadata={
                'sender': alert_data.get('sender'),
                'alert_type': alert_data.get('type')
            }
        )
        
        self.db.add(alert)
        self.db.commit()
        
        # Create processing task in queue
        task = QueueTask(
            task_json={
                'action': 'process_alert',
                'alert_id': alert.id
            },
            status='pending',
            agent_type='triage',
            priority=1
        )
        
        self.db.add(task)
        self.db.commit()
        
        return alert

    def _calculate_priority(self, severity: IncidentSeverity) -> int:
        """Calculate task priority based on incident severity."""
        priority_map = {
            IncidentSeverity.LOW: 1,
            IncidentSeverity.MEDIUM: 2,
            IncidentSeverity.HIGH: 3
        }
        return priority_map.get(severity, 1)