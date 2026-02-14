import pickle
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load precomputed embeddings
# -------------------------
EMBEDDINGS_FILE = os.path.join("data", "data_embeddings", "bfsi_alpaca_embeddings.pkl")
DATASET_FOLDER = os.path.join("data", "bfsi_alpaca_dataset_v1")

with open(EMBEDDINGS_FILE, 'rb') as f:
    embeddings = pickle.load(f)

# Convert embeddings to matrix and maintain intent_id order
intent_ids = list(embeddings.keys())
embedding_matrix = np.array([embeddings[i] for i in intent_ids])

# Load all JSON dataset entries into a dict keyed by intent_id
dataset = {}
for file in os.listdir(DATASET_FOLDER):
    if file.endswith(".json"):
        with open(os.path.join(DATASET_FOLDER, file), 'r', encoding='utf-8') as f:
            entries = json.load(f)
        for e in entries:
            dataset[e['intent_id']] = e['output']

# Load a sentence transformer model for query embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Similarity search function
# -------------------------
def get_best_match(query):
    query_vec = model.encode(query).reshape(1, -1)
    sims = cosine_similarity(query_vec, embedding_matrix)[0]
    top_idx = sims.argmax()
    return dataset[intent_ids[top_idx]]
