from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

Base = declarative_base()

class IncidentSeverity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    role_id = Column(Integer, ForeignKey('roles.id'))
    department = Column(String)
    attributes = Column(JSON)  # For ABAC
    created_at = Column(DateTime, default=datetime.utcnow)
    
    role = relationship("Role", back_populates="users")

class Role(Base):
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    permissions = Column(JSON)
    
    users = relationship("User", back_populates="role")

class IncidentLog(Base):
    __tablename__ = 'incident_logs'
    
    id = Column(Integer, primary_key=True)
    incident_json = Column(JSON)
    severity = Column(Enum(IncidentSeverity))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    metadata = Column(JSON)  # For ABAC and tagging

class SystemMetric(Base):
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metrics_json = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class AlertEmail(Base):
    __tablename__ = 'alert_emails'
    
    id = Column(Integer, primary_key=True)
    email_payload_json = Column(JSON)
    processed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class QueueTask(Base):
    __tablename__ = 'queue'
    
    id = Column(Integer, primary_key=True)
    task_json = Column(JSON)
    status = Column(Enum(TaskStatus))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    agent_type = Column(String)  # Type of agent to handle the task
    priority = Column(Integer, default=0)
    metadata = Column(JSON)

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    
    id = Column(Integer, primary_key=True)
    content = Column(JSON)
    embedding_id = Column(String)  # Reference to vector DB
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    metadata = Column(JSON)  # For ABAC and tagging

def init_db(db_url='sqlite:///data/sqlite/incident_management.db'):
    """Initialize the database and create all tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine