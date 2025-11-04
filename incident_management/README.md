# AI-Driven Incident Management System

A comprehensive incident management system leveraging AI for automated detection, classification, and resolution of incidents.

## Architecture Overview

### 1. Data Layer
- SQLite for structured data
- Vector DB for embeddings and semantic search
- Logical data separation via namespaces and metadata

### 2. Application Layer
- FastAPI backend services
- RESTful APIs for ingestion and retrieval
- Microservice Controller Plane (MCP) agents

### 3. AI Models
- BERT for classification
- Embedding models for semantic search
- LLMs for content generation
- Reinforcement learning for optimization

### 4. Orchestration Layer
- Crew AI for agent management
- Queue-based task processing
- Event-driven architecture

### 5. UI Layer
- Streamlit dashboards
- Role-based access control
- Real-time monitoring

## Project Structure
```
incident_management/
├── data/                    # Data storage and schemas
│   ├── sqlite/             # SQLite database
│   └── vector_db/          # Vector database files
├── src/                    # Source code
│   ├── auth/              # Authentication & RBAC/ABAC
│   ├── data/              # Data ingestion & storage
│   ├── models/            # AI models & embeddings
│   ├── agents/            # AI agent implementations
│   ├── orchestration/     # Task orchestration
│   ├── api/              # FastAPI application
│   └── ui/               # Streamlit dashboard
├── tests/                 # Test suite
└── docs/                 # Documentation
```

## Key Features

1. Data Ingestion & Storage
- Multi-source data ingestion
- Structured and unstructured data handling
- Efficient storage and retrieval

2. User Authentication & Access Control
- Role-Based Access Control (RBAC)
- Attribute-Based Access Control (ABAC)
- Fine-grained permissions

3. Incident Processing Pipeline
- Automated detection and logging
- AI-powered severity classification
- Semantic chunking and embedding
- RAG-based knowledge retrieval

4. Agent Orchestration
- Task queue management
- Agent coordination
- Microservice integration

5. AI Agents
- Triage Agent
- Diagnosis Agent
- Remediation Agent
- Reporting Agent
- Ticketing Agent

6. RAG System
- Context-aware retrieval
- Secure data access
- Intelligent response generation

7. Feedback & Learning
- Continuous improvement
- Knowledge base updates
- Performance monitoring

## Setup & Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Initialize the databases:
```bash
python scripts/init_db.py
```

5. Start the services:
```bash
# Start API server
uvicorn src.api.main:app --reload

# Start Streamlit dashboard
streamlit run src.ui.dashboard:main
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License