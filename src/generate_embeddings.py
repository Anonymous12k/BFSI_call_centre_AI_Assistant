from sentence_transformers import SentenceTransformer
import pickle, json, os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

dataset_folder = 'data/bfsi_alpaca_dataset_v1/'
embeddings = {}

# Loop through all JSON files
for file in os.listdir(dataset_folder):
    if file.endswith('.json'):
        with open(os.path.join(dataset_folder, file), 'r', encoding='utf-8') as f:
            entries = json.load(f)
        for e in entries:
            text = e['instruction'] + " " + e.get('input', '')
            embeddings[e['intent_id']] = model.encode(text)

# Save embeddings
os.makedirs('data/data_embeddings', exist_ok=True)
with open('data/data_embeddings/bfsi_alpaca_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Embeddings generated and saved!")
