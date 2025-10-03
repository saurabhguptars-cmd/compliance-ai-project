# =========================
# Step 2 - Compliance Q&A Engine
# =========================

import torch, numpy as np, pandas as pd, json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -------------------------------
# Load embeddings and clauses from Step 1
# -------------------------------
clauses = pd.read_csv("data/processed/clauses_sample.csv")["clause"].tolist()
embeddings = np.load("data/processed/corpus_embeddings.npy")
embeddings = torch.tensor(embeddings)

print(f"Loaded {len(clauses)} clauses and embeddings.")

# Reload the same embedder used in Step 1
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Semantic search function
# -------------------------------
def search(query, top_k=3):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)[0]
    results = []
    for h in hits:
        results.append({
            "score": float(h["score"]),
            "clause": clauses[h["corpus_id"]]
        })
    return results

# -------------------------------
# Summarizer and Simplifier
# -------------------------------
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
simplifier = pipeline("text2text-generation", model="google/flan-t5-small", device=device)

def explain_clause(text):
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
    prompt = "Simplify the following legal clause into 3-4 clear action items:\n\n" + text
    items = simplifier(prompt, max_length=200)[0]['generated_text']
    return summary, items

# -------------------------------
# Example Query
# -------------------------------
query = "Where must customer data be stored?"
results = search(query, top_k=3)

print(f"\nQuery: {query}\n")
for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Clause: {r['clause']}\n")
    summary, items = explain_clause(r['clause'])
    print("Summary:", summary)
    print("Action Items:", items)
    print("\n---\n")
