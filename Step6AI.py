# =========================
# Step 6: Dynamic Live Web Page Compliance Check
# =========================

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Device setup for embeddings
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device set to use", device)

# Load Step 1 clauses and embedder
clauses_df = pd.read_csv("data/processed/clauses_sample.csv")
clauses = clauses_df['clause'].tolist()
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Define URLs to check
urls = {
    "homepage": "https://www.bankofamerica.com/",
    "privacy": "https://www.bankofamerica.com/privacy/",
    "consumer_privacy": "https://www.bankofamerica.com/privacy/consumer-privacy/",
    "terms_of_service": "https://www.bankofamerica.com/online-banking/terms-of-service/",
    "security": "https://www.bankofamerica.com/security/"
}

# Smart contract rules
smart_contracts = {
    "data_location": "All customer data must be stored within the European Union",
    "sensitive_access": "Access to sensitive data must be restricted to authorized personnel"
}

# -------------------------------
# Function to scrape text from URL
# -------------------------------
def scrape_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # Extract all text and limit to first 5000 characters
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000]
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# -------------------------------
# Function to extract metrics using AI/NLP
# -------------------------------
def extract_metrics(page_text):
    # Simplified AI/NLP extraction: check for keywords in text
    data_location = "EU" if "European Union" in page_text else "US"
    sensitive_access = "authorized_only" if any(
        x in page_text.lower() for x in ["authorized", "employees only", "restricted"]
    ) else "everyone"
    return {"data_location": data_location, "sensitive_access": sensitive_access}

# -------------------------------
# Function to evaluate compliance
# -------------------------------
def evaluate_compliance(page_text, metrics):
    alerts = []
    suggested_changes = []

    for key, rule_text in smart_contracts.items():
        if key in metrics:
            metric_value = metrics[key]
            # Semantic similarity using embeddings
            rule_emb = embedder.encode([rule_text], convert_to_tensor=True)
            metric_emb = embedder.encode([str(metric_value)], convert_to_tensor=True)
            score = util.cos_sim(rule_emb, metric_emb).item()
            
            if score < 0.4:
                alerts.append(f"⚠ Non-compliance on {key}: value='{metric_value}' vs rule='{rule_text}' (score={score:.2f})")
                suggested_changes.append(f"Change '{key}' to comply with: '{rule_text}'")
            else:
                alerts.append(f"✅ Compliant on {key}: value='{metric_value}' (score={score:.2f})")
    return alerts, suggested_changes

# -------------------------------
# Run monitoring for all URLs
# -------------------------------
for page_name, url in urls.items():
    print(f"\n--- Monitoring {page_name} ({url}) ---")
    page_text = scrape_text(url)
    print(f"Extracted text from {page_name}: {len(page_text)} characters")
    
    metrics = extract_metrics(page_text)
    print("Extracted metrics:", metrics)
    
    alerts, suggested_changes = evaluate_compliance(page_text, metrics)
    
    for a in alerts:
        print(a)
    if suggested_changes:
        print("\nSuggested Actions:")
        for s in suggested_changes:
            print("-", s)
