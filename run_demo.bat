@echo off
echo Starting RAG Pipeline Demo with Hugging Face models...

:: Check if HUGGINGFACE_API_TOKEN is set
if "%HUGGINGFACE_API_TOKEN%"=="" (
    echo WARNING: HUGGINGFACE_API_TOKEN is not set!
    echo Please set it using:
    echo set HUGGINGFACE_API_TOKEN=your_token_here
    pause
)

:: Run the Streamlit app
echo Starting Streamlit server...
C:\Users\PRADHAN\AppData\Local\Programs\hackathon-build\.venv\Scripts\streamlit.exe run ^
    "C:\Users\PRADHAN\AppData\Local\Programs\hackathon-build\rag_pipeline\demo\run_huggingface_demo.py"

pause