# Hackathon Build — Data Ingest, Synthetic Generator, Clustering & LLM Analysis

This workspace contains a compact end-to-end demo pipeline for:

- Generating YAML-driven synthetic datasets (`synthetic_data.py`)
- Centralized ingestion via a pluggable `DataIngestor` (`data_ingest.py`)
- Preprocessing, PCA, clustering and feature-engineering (`full_pipeline_demo.py`)
- Advanced statistical + PCA analysis with an LLM-style summarizer (`advanced_pca_analysis.py` / `visualize_pca.py`)
- Small demos: Chroma vector demo, PDF example, in-memory IoT stream replacement for Kafka

Purpose
-------
This repository is a runnable sandbox that shows how to:

- Produce synthetic (realistic, mixed-type) data from a YAML schema
- Persist the data to CSV, re-ingest and automatically extract numeric features
- Clean the data (imputation, scaling), reduce dimensionality (PCA), and cluster
- Generate engineered features (one-hot, cluster id, distance-to-centroid, PCA components)
- Produce human- and LLM-friendly analysis and recommendations

Key files
---------
- `data_ingest.py` — DataIngestor class with built-in loaders (csv, sqlite, json, texts, sample). Extensible via `register_loader()`.
- `synthetic_data.py` — YAML-driven synthetic data generator. Use `generate_from_yaml()` or `generate_from_schema()`.
- `sample_schema.yaml` — example schema used by demos.
- `run_synthetic_ingest.py` — generates synthetic data from YAML, ingests using `DataIngestor` and writes `synthetic_output.csv`.
- `test_csv_ingest.py` — small script to load `synthetic_output.csv` through `DataIngestor` (CSV loader).
- `full_pipeline_demo.py` — end-to-end pipeline: generate → ingest → impute/scale → PCA → cluster → feature engineering → `enhanced_data.csv`.
- `advanced_pca_analysis.py` — performs univariate & bivariate analysis, correlation, PCA, cluster summaries and calls `llm_agent` for an expert-style summary.
- `visualize_pca.py` / `advanced_pca_analysis.py` — plotting helpers and PCA visualizations (`pca_analysis.png`, `advanced_analysis.png`).
- `llm_agent.py` — wrapper that calls OpenAI (if `OPENAI_API_KEY` present), Hugging Face Inference (if configured), or returns a safe deterministic summary.
- `vector_demo_chroma.py`, `nw.py`, `nw_demo.py`, `jal_nokafka.py`, `create_pdf.py` — additional demos and helpers used while building the project.

Quick start (Windows / PowerShell)
---------------------------------
1. Create & activate virtualenv (if not already):

```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

3. Generate synthetic data and ingest it (creates `synthetic_output.csv`):

```powershell
.\.venv\Scripts\python.exe run_synthetic_ingest.py
```

4. Run the full pipeline (generates `raw_data.csv`, `enhanced_data.csv`, plots):

```powershell
.\.venv\Scripts\python.exe full_pipeline_demo.py
```

5. Run advanced analysis and get an LLM-style summary (uses keys if set):

```powershell
.\.venv\Scripts\python.exe advanced_pca_analysis.py
```

Environment variables (optional)
--------------------------------
- `OPENAI_API_KEY` — enable OpenAI ChatCompletion usage for richer LLM summaries.
- `HUGGINGFACE_API_KEY` — enable Hugging Face Inference API fallback.
- `HF_MODEL` — model repo for HF inference (default: `google/flan-t5-large`).
- `HF_HOME`, `TRANSFORMERS_CACHE` — optional to redirect model caches to another drive (useful on Windows C: space constraints).

What the pipeline writes
------------------------
- `synthetic_output.csv` — YAML-generated CSV from `run_synthetic_ingest.py`
- `raw_data.csv` — raw synthetic data (from `full_pipeline_demo.py`)
- `enhanced_data.csv` — engineered dataset for modeling (one-hot, cluster, distances, PCA)
- `pca_variance.png`, `silhouette_scores.png`, `pca_analysis.png`, `advanced_analysis.png` — visual artifacts created by demos
- `analysis_llm_summary.txt` — (optional) place where you can save LLM output if you enable that feature

Notes & gotchas
----------------
- sentence-transformers: The text->embedding loader requires `sentence-transformers`. The loader in `data_ingest.py` raises an informative error if the package is missing.
- TensorFlow/Keras: When running some demos you may see TF/Keras warnings. If you run into compatibility issues, ensure `tf-keras` and compatible `tensorflow` are installed for your environment.
- Chroma: The Chroma demo includes a persistent vs in-memory fallback. If your installed `chromadb` version requires a migration, the demo falls back to an in-memory client and prints migration guidance.
- LLM usage: Calling OpenAI or Hugging Face will make network calls and may incur costs. If keys are not set, the repo uses a safe template summary so the pipeline remains fully offline-capable.

Best practices & next steps
-------------------------
- Use the `synthetic_data.py` YAML schema to iterate quickly on feature shapes and missingness scenarios before connecting to production data sources.
- For production ingestion, extend `DataIngestor.register_loader()` to add connectors (S3, Postgres, Kafka, Qdrant, etc.).
- Add unit tests for `DataIngestor` and `synthetic_data` (happy path + edge cases). A tiny `tests/` folder with pytest tests will make CI validation easier.
- If you enable LLMs, consider adding prompt templates and a token-aware summarizer to avoid hitting token limits.

License & attribution
---------------------
This code is provided as-is for demo and prototyping purposes. If you reuse external models, follow their licenses and usage terms (OpenAI, Hugging Face, sentence-transformers, Chroma).

Questions or changes
--------------------
If you'd like, I can:

- Add a `README_RUN.md` with copy-paste PowerShell snippets for common tasks
- Add a `Makefile` / `tasks.json` for VS Code to run the demos with one click
- Add unit tests and GitHub Actions for CI

Pick one of the next steps and I will implement it.
