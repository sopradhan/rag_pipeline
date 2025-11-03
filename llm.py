import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

# Login to Hugging Face
login(token=HF_TOKEN)

model_name = "meta-llama/Llama-2-7b-chat-hf"

# ✅ Check access without downloading weights
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
print("✅ Tokenizer OK — access confirmed!")

# Optional: lightweight config check
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name, use_auth_token=True)
print("✅ Config OK — you have access to the model.")
