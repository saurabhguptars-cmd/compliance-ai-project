# -------------------------------
# Required Libraries
# -------------------------------
!pip install sentence-transformers transformers pandas requests beautifulsoup4 tqdm

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Step 1: Collect Legal Documents & Domain Docs
# -------------------------------

# Example legal document (public)
legal_doc_url = "https://www.sec.gov/about/laws.shtml"  # US Securities Laws page
response = requests.get(legal_doc_url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract text paragraphs
legal_paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]

# Example Bank of America app documentation (publicly available)
boa_doc_urls = [
    "https://www.bankofamerica.com/mobile-banking/",
    "https://www.bankofamerica.com/deposits/online-banking-features/"
]

app_paragraphs = []
for url in boa_doc_urls:
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    paras = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]
    app_paragraphs.extend(paras)

# Combine into DataFrames
legal_df = pd.DataFrame({"document": "Legal", "text": legal_paragraphs})
app_df = pd.DataFrame({"document": "BankApp", "text": app_paragraphs})

all_docs_df = pd.concat([legal_df, app_df], ignore_index=True)
print(f"Total paragraphs collected: {len(all_docs_df)}")

# -------------------------------
# Step 2: Embedding and Contextual Understanding
# -------------------------------

# Load pre-trained sentence-transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
all_docs_df['embedding'] = all_docs_df['text'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# -------------------------------
# Step 3: Semantic Search / Contextual Matching
# -------------------------------

# Example: Check which legal clauses relate to app functionality
query = "Privacy policy adherence for user data in mobile banking app"
query_emb = model.encode(query, convert_to_tensor=True)

# Compute similarity
all_docs_df['similarity'] = all_docs_df['embedding'].apply(lambda x: util.cos_sim(query_emb, x).item())

# Sort by most relevant
relevant_docs = all_docs_df.sort_values(by='similarity', ascending=False).head(10)

# Display top relevant paragraphs
for idx, row in relevant_docs.iterrows():
    print(f"[{row['document']}] Similarity: {row['similarity']:.3f}\n{row['text']}\n{'-'*80}")

# -------------------------------
# Step 4: Human-Readable Summary (optional)
# -------------------------------
# For each relevant paragraph, you can use a summarization model (transformers) to convert
# legal clauses into simplified statements.
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

relevant_docs['summary'] = relevant_docs['text'].apply(
    lambda x: summarizer(x, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
)

# Print summaries
for idx, row in relevant_docs.iterrows():
    print(f"[{row['document']}] Summary: {row['summary']}\n{'='*80}")
