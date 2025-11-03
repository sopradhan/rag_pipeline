# Comprehensive RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline supporting multiple data sources, hybrid classification, clustering, access control, and content guardrails.

## Features

- Multi-source data ingestion (DB, CSV, PDF, web, etc.)
- Hybrid classification (ML + rules)
- Vector embeddings & clustering
- RBAC/ABAC access control
- Content guardrails & PII detection
- Interactive visualization dashboard

## Getting Started

1. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run demo:
```bash
python -m rag_pipeline.demo
```

4. Launch dashboard:
```bash
streamlit run rag_pipeline/dashboard/app.py
```

## Project Structure

```
rag_pipeline/
├── data/               # Sample data & schemas
├── rag_pipeline/       # Main package
│   ├── ingestion/     # Data source adapters
│   ├── classify/      # Classification system
│   ├── embed/         # Embedding & clustering
│   ├── security/      # Access control & guardrails
│   ├── api/          # FastAPI backend
│   └── dashboard/    # Streamlit frontend
├── tests/            # Test suite
└── notebooks/        # Example notebooks
```

## Documentation

See [docs/](docs/index.md) for detailed documentation.

## License

MIT