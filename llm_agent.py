"""llm_agent.py
Lightweight wrapper to call an LLM for analysis interpretation.

Behavior:
- If OPENAI_API_KEY is set and `openai` package is installed, use OpenAI ChatCompletion (gpt-3.5-turbo).
- Else if HUGGINGFACE_API_KEY is set, call Hugging Face Inference API (text generation) using requests.
- Else, return a safe, deterministic template-based summary (no network calls).

The module exposes `summarize_insights(text)` which returns a string with LLM-style recommendations.
"""
from typing import Optional
import os
import json
import textwrap

def _safe_template_summary(analysis_text: str) -> str:
    """Return a deterministic summary when no LLM key is available."""
    # Keep it short and actionable
    lines = [l.strip() for l in analysis_text.splitlines() if l.strip()]
    top_lines = lines[:20]
    summary = ["LLM Unavailable — returning deterministic summary:"]
    summary.append("Top observations and action items:")
    # pick sentences mentioning PCA, cluster or missing
    for l in top_lines:
        if any(k in l.lower() for k in ('pca', 'cluster', 'missing', 'imput', 'variance', 'correl')):
            summary.append(f"- {l}")
    # fallback: include first 5 lines
    if len(summary) <= 2:
        summary.extend([f"- {l}" for l in top_lines[:5]])
    summary.append('Suggested next steps: impute missing values, standardize features, evaluate models (RF/XGBoost), and consider cluster-based features.')
    return "\n".join(summary)


def summarize_insights(analysis_text: str, model: Optional[str] = None) -> str:
    """Summarize and expand analysis_text using an LLM if available.

    Tries OpenAI -> HuggingFace -> deterministic fallback.
    """
    # Prefer OpenAI if key is set
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            # Build a prompt instructing the model to act like a senior data scientist
            system = "You are a senior data scientist. Read the analysis and provide a concise, actionable interpretation, list 3-5 recommendations, potential pitfalls, and next experiments to run."
            user_prompt = f"Analysis:\n{analysis_text}\n\nProvide an executive summary, recommendations, and next steps. Keep it brief and numbered."
            resp = openai.ChatCompletion.create(
                model=model or 'gpt-3.5-turbo',
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                max_tokens=400,
                temperature=0.2,
            )
            return resp['choices'][0]['message']['content'].strip()
        except Exception as e:
            # If anything fails, fall through to next option
            err = str(e)

    # Try Hugging Face Inference API if key available
    hf_key = os.environ.get('HUGGINGFACE_API_KEY')
    if hf_key:
        try:
            import requests
            repo = os.environ.get('HF_MODEL', 'google/flan-t5-large')
            prompt = f"Read the analysis below and provide a concise executive summary (3 bullets) and 3 next steps:\n\n{analysis_text}"
            headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.2}}
            url = f"https://api-inference.huggingface.co/models/{repo}"
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            out = r.json()
            # HuggingFace inference returns a list of dicts or a dict depending on model
            if isinstance(out, list):
                text = out[0].get('generated_text') or out[0].get('text') or str(out[0])
            elif isinstance(out, dict):
                text = out.get('generated_text') or out.get('text') or json.dumps(out)
            else:
                text = str(out)
            return text.strip()
        except Exception:
            pass

    # Fallback deterministic summary
    return _safe_template_summary(analysis_text)


def suggest_hyperparams(analysis_text: str, model: Optional[str] = None) -> dict:
    """Ask an LLM for numeric hyperparameter suggestions. Returns a dict with keys:
    - k: suggested number of clusters (int)
    - imputer: 'median'|'mean'|'knn'|'iterative'
    - scaler: 'robust'|'standard'|'minmax'

    Falls back to deterministic heuristic when no LLM available.
    """
    # Try OpenAI path
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            prompt = (
                "You are a ML engineer. Given the analysis below, suggest numeric hyperparameters for clustering as JSON. "
                "Return only JSON. Fields: k (int), imputer (one of median, mean, knn, iterative), scaler (robust, standard, minmax).\n\n"
                f"Analysis:\n{analysis_text}\n\nJSON:\n"
            )
            resp = openai.ChatCompletion.create(
                model=model or 'gpt-3.5-turbo',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.0,
            )
            text = resp['choices'][0]['message']['content'].strip()
            # Try to parse JSON from response
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                jtxt = m.group(0)
                return json.loads(jtxt)
        except Exception:
            pass

    # Try HF inference
    hf_key = os.environ.get('HUGGINGFACE_API_KEY')
    if hf_key:
        try:
            import requests
            repo = os.environ.get('HF_MODEL', 'google/flan-t5-large')
            prompt = (
                "Given the analysis below, output a JSON object with fields: k (int), imputer (median|mean|knn|iterative), scaler (robust|standard|minmax). Only output JSON.\n\n"
                f"Analysis:\n{analysis_text}"
            )
            headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 120, "temperature": 0.0}}
            url = f"https://api-inference.huggingface.co/models/{repo}"
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            out = r.json()
            if isinstance(out, list):
                text = out[0].get('generated_text') or out[0].get('text') or str(out[0])
            elif isinstance(out, dict):
                text = out.get('generated_text') or out.get('text') or json.dumps(out)
            else:
                text = str(out)
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass

    # Deterministic heuristic fallback
    # Simple rules: set k = clamp(n_features*2, 3, 12), pick imputer based on missingness heuristics in text
    # Try to extract approximate n_features and missing% from analysis_text
    try:
        import re
        m = re.search(r"(\d+) features", analysis_text)
        n_features = int(m.group(1)) if m else 3
    except Exception:
        n_features = 3
    k = max(3, min(12, n_features * 2))
    imputer = 'median'
    if 'missing' in analysis_text.lower():
        # if text mentions >5% missing prefer knn
        if re.search(r"(\d+\.?\d*)%", analysis_text):
            imputer = 'knn'
        else:
            imputer = 'median'
    scaler = 'robust'
    return {'k': int(k), 'imputer': imputer, 'scaler': scaler}


if __name__ == '__main__':
    # quick local test when run directly
    sample = 'PCA: PC1 explains 40%. Missing: sensor1 5%. Clustering found 4 clusters.'
    print(summarize_insights(sample))
