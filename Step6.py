# =========================
# Step 6: Live Website Scraping + AI Compliance Monitoring
# =========================

import requests
from bs4 import BeautifulSoup
import json
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load Step 1 embeddings and clauses
# -------------------------------
import numpy as np
import pandas as pd

clauses_df = pd.read_csv("data/processed/clauses_sample.csv")
clauses = clauses_df['clause'].tolist()
corpus_embeddings = np.load("data/processed/corpus_embeddings.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# URLs to scrape
# -------------------------------
urls = {
    "homepage": "https://www.bankofamerica.com/",
    "privacy": "https://www.bankofamerica.com/privacy/"
}

# -------------------------------
# Scrape pages and extract text
# -------------------------------
pages_text = {}
for page_name, url in urls.items():
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)[:5000]  # first 5000 chars
        pages_text[page_name] = text
        print(f"Extracted text from {page_name}: {len(text)} characters")
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

# -------------------------------
# Define smart contract rules
# -------------------------------
smart_contracts = {
    "data_location": "All customer data must be stored within the European Union",
    "sensitive_access": "Access to sensitive data must be restricted to authorized personnel"
}

# -------------------------------
# Monitoring function
# -------------------------------
def monitor_page(page_name, text):
    print(f"\nTop clauses matched for {page_name}:")
    text_emb = embedder.encode([text], convert_to_tensor=True)
    hits = util.semantic_search(text_emb, corpus_embeddings, top_k=3)[0]
    for h in hits:
        print("-", clauses[h["corpus_id"]])
    
    print(f"\nMonitoring {page_name}...")
    alerts = []
    suggested_changes = []
    
    # Example metrics (simulated from text, can enhance later)
    metrics = {
        "data_location": "US",             # default assumed
        "sensitive_access": "everyone"     # default assumed
    }
    
    for key, rule_text in smart_contracts.items():
        if key in metrics:
            metric_value = metrics[key]
            rule_emb = embedder.encode([rule_text], convert_to_tensor=True)
            metric_emb = embedder.encode([str(metric_value)], convert_to_tensor=True)
            score = util.cos_sim(rule_emb, metric_emb).item()
            
            if score < 0.4:
                alert_msg = f"⚠ Non-compliance on {key}: value='{metric_value}' vs rule='{rule_text}' (score={score:.2f})"
                suggested_change = f"Change '{key}' of {page_name} to comply with: '{rule_text}'"
                alerts.append(alert_msg)
                suggested_changes.append(suggested_change)
    
    if not alerts:
        print("✅ All metrics compliant")
    else:
        print("\n".join(alerts))
        print("\nSuggested Actions:")
        for action in suggested_changes:
            print("-", action)
    
    # Save report
    report = {
        "page_name": page_name,
        "alerts": alerts,
        "suggested_changes": suggested_changes
    }
    with open(f"data/processed/{page_name}_compliance_report.json", "w") as f:
        json.dump(report, f, indent=2)

# -------------------------------
# Run monitoring on all pages
# -------------------------------
for page, text in pages_text.items():
    monitor_page(page, text)

print("\n✅ Live public data compliance check completed.")
