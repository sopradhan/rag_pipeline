import pytest
import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.models import Base, IncidentLog, User, Role
from src.data.ingestion import DataIngester
from src.auth.security import AuthManager, AccessControl
from src.vector_db.client import VectorDBClient
from src.ai.classification import classify_severity
from scripts.generate_mock_data import MockDataGenerator

# Test configuration
TEST_DB_URL = "sqlite:///test.db"
TEST_VECTOR_DB_URL = "localhost"
TEST_VECTOR_DB_PORT = 6333

@pytest.fixture(scope="session")
def db_engine():
    """Create test database engine."""
    engine = create_engine(TEST_DB_URL)
    Base.metadata.create_all(engine)
    yield engine
    os.remove("test.db")

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create new database session for each test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture(scope="session")
def vector_db():
    """Initialize vector database client."""
    return VectorDBClient(
        url=TEST_VECTOR_DB_URL,
        port=TEST_VECTOR_DB_PORT
    )

@pytest.fixture(scope="session")
def mock_data():
    """Generate mock incident data."""
    generator = MockDataGenerator()
    return generator.generate_incidents(10)

class TestDataIngestion:
    async def test_incident_ingestion(self, db_session, vector_db, mock_data):
        """Test incident data ingestion."""
        ingester = DataIngester(db_session, vector_db)
        incident = mock_data[0]
        
        # Test ingestion
        incident_log = await ingester.ingest_incident(incident)
        assert incident_log.id is not None
        assert incident_log.severity == incident["severity"]
        
        # Verify database entry
        db_incident = db_session.query(IncidentLog).filter_by(id=incident_log.id).first()
        assert db_incident is not None
        assert db_incident.incident_json["type"] == incident["type"]
        
        # Verify vector storage
        vector_id = db_incident.metadata["vector_id"]
        assert vector_id is not None

class TestAuthentication:
    def test_user_creation(self, db_session):
        """Test user creation and authentication."""
        auth_manager = AuthManager(db_session)
        
        # Create test role
        role = Role(name="test_role", permissions=["view_incidents"])
        db_session.add(role)
        db_session.commit()
        
        # Create test user
        password = "test_password"
        user = User(
            username="test_user",
            email="test@example.com",
            password_hash=auth_manager.get_password_hash(password),
            role_id=role.id
        )
        db_session.add(user)
        db_session.commit()
        
        # Test authentication
        authenticated = auth_manager.verify_password(password, user.password_hash)
        assert authenticated is True

class TestAccessControl:
    def test_rbac_permissions(self, db_session):
        """Test RBAC permission checking."""
        access_control = AccessControl(db_session)
        
        # Create test role and user
        role = Role(name="admin", permissions=["view_incidents", "edit_incidents"])
        db_session.add(role)
        db_session.commit()
        
        user = User(
            username="admin_user",
            email="admin@example.com",
            role_id=role.id
        )
        db_session.add(user)
        db_session.commit()
        
        # Test permissions
        assert access_control.check_rbac_permission(user, "view_incidents") is True
        assert access_control.check_rbac_permission(user, "delete_incidents") is False

    def test_abac_permissions(self, db_session):
        """Test ABAC permission checking."""
        access_control = AccessControl(db_session)
        
        # Create user with attributes
        user = User(
            username="dept_user",
            email="dept@example.com",
            attributes={
                "department": "IT",
                "security_clearance": 2,
                "projects": ["project_a", "project_b"]
            }
        )
        db_session.add(user)
        db_session.commit()
        
        # Test resource access
        resource_attrs = {
            "department": "IT",
            "required_clearance": 2,
            "project": "project_a"
        }
        assert access_control.check_abac_permission(user, resource_attrs) is True
        
        # Test restricted access
        restricted_attrs = {
            "department": "Finance",
            "required_clearance": 3,
            "project": "project_c"
        }
        assert access_control.check_abac_permission(user, restricted_attrs) is False

class TestAIComponents:
    async def test_severity_classification(self, mock_data):
        """Test incident severity classification."""
        incident = mock_data[0]
        severity = await classify_severity(incident["description"])
        assert severity in ["LOW", "MEDIUM", "HIGH"]

    async def test_vector_storage_retrieval(self, vector_db, mock_data):
        """Test vector storage and retrieval."""
        incident = mock_data[0]
        
        # Store vector
        vector_id = await vector_db.store(
            [0.1] * 768,  # Mock embedding
            metadata={
                "incident_type": incident["type"],
                "severity": incident["severity"]
            }
        )
        assert vector_id is not None
        
        # Search vectors
        results = await vector_db.search(
            [0.1] * 768,  # Mock query vector
            limit=1
        )
        assert len(results) > 0
        assert results[0]["metadata"]["incident_type"] == incident["type"]

if __name__ == "__main__":
    pytest.main(["-v"])