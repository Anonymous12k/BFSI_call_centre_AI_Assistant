# src/rag_retriever.py

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Sensitive keywords (guardrails)
# -------------------------
SENSITIVE_KEYWORDS = [
    "password", "pin", "credit card number", "account number",
    "social security", "aadhaar", "personal info", "sensitive"
]

def is_safe_query(query):
    """Return False if the query contains sensitive or unsafe information."""
    query_lower = query.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in query_lower:
            return False
    return True

# -------------------------
# Load all JSON files in the RAG folder
# -------------------------
RAG_FOLDER = os.path.join("data", "rag_docs")
rag_entries = []

if not os.path.exists(RAG_FOLDER):
    os.makedirs(RAG_FOLDER)

json_files = [f for f in os.listdir(RAG_FOLDER) if f.endswith(".json")]
for file in json_files:
    path = os.path.join(RAG_FOLDER, file)
    with open(path, 'r', encoding='utf-8') as f:
        try:
            entries = json.load(f)
            # Validate each entry has 'keywords' and 'answer'
            for e in entries:
                if "keywords" in e and "answer" in e:
                    rag_entries.append(e)
        except Exception as err:
            print(f"Error loading {file}: {err}")

print(f"Loaded {len(rag_entries)} RAG entries from {len(json_files)} JSON files.")

# -------------------------
# Prepare texts for embeddings
# -------------------------
rag_texts = []
for entry in rag_entries:
    keywords = entry.get("keywords", [])
    answer = entry.get("answer", "")
    combined_text = " ".join(keywords) + " " + answer
    rag_texts.append(combined_text)

# -------------------------
# Compute embeddings
# -------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
rag_embeddings = embedding_model.encode(rag_texts, convert_to_numpy=True)

# -------------------------
# RAG retrieval function
# -------------------------
def retrieve_rag_answer(query, similarity_threshold=0.6):
    """
    Returns the most relevant answer from any RAG JSON file.
    Uses semantic similarity with embeddings and respects sensitive keyword guardrails.
    """
    if not query.strip() or not is_safe_query(query):
        return "Sorry, I cannot process sensitive or unsafe information."

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_embedding, rag_embeddings)[0]

    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= similarity_threshold:
        return rag_entries[best_idx]["answer"]
    return None

# -------------------------
# Test CLI
# -------------------------
if __name__ == "__main__":
    print("RAG Retriever CLI (type 'exit' to quit)")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            break
        answer = retrieve_rag_answer(query)
        if answer:
            print(f"\nResponse: {answer}")
        else:
            print("\nResponse: No relevant information found.")
