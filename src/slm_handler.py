# src/slm_handler.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------
# Load GPT-Neo 125M directly from HF (prototype)
# -------------------------
MODEL_NAME = "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(query, max_new_tokens=150):
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(query, "").strip()
    return response

if __name__ == "__main__":
    print("SLM Test CLI (type 'exit' to quit)")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            break
        print(generate_response(query))
