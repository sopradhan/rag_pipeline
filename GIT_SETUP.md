# Git Setup and Issue Resolution Guide

## Initial Setup Issues

1. Git not recognized in terminal
```powershell
git : The term 'git' is not recognized as the name of a cmdlet...
```

### Resolution
- Git was installed in `D:\Git`
- Used full path to Git executable: `D:\Git\cmd\git.exe`

## GitHub Token Security Issue

### Problem
GitHub's secret scanning detected Hugging Face tokens in the codebase:
```
remote: error: GH013: Repository rule violations found for refs/heads/main
remote: - Push cannot contain secrets
```

### Files Containing Sensitive Information
1. egrid.py
2. hackathon1.py
3. llm-test.py
4. llm.py

### Resolution Steps

1. Created `.env.example` for template
2. Modified code to use environment variables
3. Added proper `.gitignore`
4. Reset Git history

## Important Git Commands Used

### Repository Setup
```powershell
# Initialize repository
D:\Git\cmd\git.exe init

# Add remote
D:\Git\cmd\git.exe remote add origin https://github.com/sopradhan/rag_pipeline.git

# Set main branch
D:\Git\cmd\git.exe branch -M main
```

### Basic Operations
```powershell
# Add files
D:\Git\cmd\git.exe add .

# Commit changes
D:\Git\cmd\git.exe commit -m "Your commit message"

# Push to GitHub
D:\Git\cmd\git.exe push -u origin main

# Force push (when needed)
D:\Git\cmd\git.exe push -f -u origin main
```

### Reset Repository
```powershell
# Remove Git folder
Remove-Item -Recurse -Force .git

# Remove remote
D:\Git\cmd\git.exe remote remove origin
```

## Best Practices Implemented

1. Environment Variables
   - Created `.env` for sensitive data
   - Added `.env.example` for documentation
   - Added `.env` to `.gitignore`

2. Code Changes
   ```python
   # Before
   hf_api_key = "your_token_here"

   # After
   import os
   from dotenv import load_dotenv
   load_dotenv()
   hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
   ```

3. Security
   - Never commit tokens directly in code
   - Use environment variables
   - Keep `.env` local only

## Git Configuration
```powershell
# Set user email
D:\Git\cmd\git.exe config --global user.email "your.email@example.com"

# Set username
D:\Git\cmd\git.exe config --global user.name "your_username"
```

## Troubleshooting

1. If secrets are detected in Git history:
   - Remove `.git` directory
   - Start fresh repository
   - Push with clean history

2. If Git is not recognized:
   - Use full path: `D:\Git\cmd\git.exe`
   - Or add Git to system PATH

## Future Reference

1. Always check for sensitive data before committing:
   ```powershell
   D:\Git\cmd\git.exe diff
   D:\Git\cmd\git.exe status
   ```

2. If you accidentally commit sensitive data:
   - Don't just remove it in a new commit
   - Reset the repository and start fresh
   - Generate new tokens/keys