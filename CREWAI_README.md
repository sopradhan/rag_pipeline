CrewAI Deployment Notes

Required environment variables (set these in your `.env` or shell):

- OPENAI_API_KEY: API key for OpenAI (if using OpenAI LLM in crewai_tools)
- QDRANT_URL: URL for your Qdrant vector DB (e.g., http://localhost:6333)
- QDRANT_API_KEY: (optional) API key for Qdrant
- HF_HOME / TRANSFORMERS_CACHE: (optional) path to HF cache
- OTHER_KEYS: Any other provider keys used by crewai adapters

Quick start

1. Install packages (if not already):

   pip install -r requirements.txt

2. Fill in your `.env` with the keys above or export them in PowerShell:

   $env:OPENAI_API_KEY="sk_xxx"
   $env:QDRANT_URL="http://localhost:6333"

3. Run the demo:

   & .venv/Scripts/python.exe nw_demo.py

Notes

- If you do not provide provider keys, `nw_demo.py` will fall back to mock implementations for safe local testing.
- Qdrant can be run locally via Docker: `docker run -p 6333:6333 qdrant/qdrant`.

Chroma (local vector DB) notes

- Chroma is a lightweight, in-process vector DB ideal for development and demos.
- Example demo `vector_demo_chroma.py` uses `sentence-transformers` and persists data to `E:/epoch_explorer/chroma` by default.
- To run the demo, install dependencies and run:

   & .venv/Scripts/python.exe vector_demo_chroma.py

You can change the `persist_directory` in the demo to any path on your system.
