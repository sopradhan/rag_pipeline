from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import timedelta

from ..auth.security import AuthManager, AccessControl
from ..data.models import get_db
from ..data.ingestion import DataIngester
from ..orchestration.agent_manager import AgentOrchestrator
from ..vector_db.client import VectorDBClient

app = FastAPI(title="Incident Management System")

# Dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_auth_manager(db: Session = Depends(get_db)):
    return AuthManager(db)

def get_access_control(db: Session = Depends(get_db)):
    return AccessControl(db)

def get_vector_db():
    return VectorDBClient()

def get_data_ingester(db: Session = Depends(get_db),
                     vector_db: VectorDBClient = Depends(get_vector_db)):
    return DataIngester(db, vector_db)

def get_orchestrator(db: Session = Depends(get_db),
                    access_control: AccessControl = Depends(get_access_control)):
    return AgentOrchestrator(db, access_control)

# Authentication endpoints
@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    user = await auth_manager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_manager.create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Incident management endpoints
@app.post("/incidents")
async def create_incident(
    incident: Dict[str, Any],
    data_ingester: DataIngester = Depends(get_data_ingester),
    auth_manager: AuthManager = Depends(get_auth_manager),
    token: str = Depends(oauth2_scheme)
):
    """Create a new incident."""
    user = await auth_manager.get_current_user(token)
    incident["created_by"] = user.username
    return await data_ingester.ingest_incident(incident)

@app.get("/incidents")
async def list_incidents(
    auth_manager: AuthManager = Depends(get_auth_manager),
    access_control: AccessControl = Depends(get_access_control),
    token: str = Depends(oauth2_scheme)
):
    """List incidents with RBAC/ABAC filtering."""
    user = await auth_manager.get_current_user(token)
    incidents = []  # Get from database
    return await access_control.filter_resources_by_permission(user, incidents)

# Task management endpoints
@app.post("/tasks")
async def create_task(
    task: Dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    auth_manager: AuthManager = Depends(get_auth_manager),
    token: str = Depends(oauth2_scheme)
):
    """Create a new task."""
    user = await auth_manager.get_current_user(token)
    task["created_by"] = user.username
    return orchestrator.add_task(task, task["agent_type"], task.get("priority", 1))

@app.get("/tasks/status")
async def get_task_status(
    task_id: int,
    auth_manager: AuthManager = Depends(get_auth_manager),
    token: str = Depends(oauth2_scheme)
):
    """Get task status."""
    user = await auth_manager.get_current_user(token)
    # Implement task status retrieval
    pass

# Agent endpoints
@app.post("/agents/process")
async def process_tasks(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    auth_manager: AuthManager = Depends(get_auth_manager),
    token: str = Depends(oauth2_scheme)
):
    """Process pending tasks in queue."""
    user = await auth_manager.get_current_user(token)
    return await orchestrator.process_queue()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)