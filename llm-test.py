import os
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

# Login to Hugging Face
login(token=HF_TOKEN)
model_name = "meta-llama/Llama-2-7b-chat-hf"



# Load tokenizer to format the prompt properly
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    device_map="auto",      # GPU if available, else CPU
    torch_dtype="auto",
    use_auth_token=HF_TOKEN # <-- required for gated repo
)

# Convert chat messages into a single prompt
messages = [{"role": "user", "content": "Who are you?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate
out = pipe(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9)
print(out[0]["generated_text"])
