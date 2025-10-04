# =========================
# Step 6: Dynamic Compliance Check with Suggestions
# =========================

!pip install -q sentence-transformers requests beautifulsoup4 pandas

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 1️⃣ Define pages to scrape
# -------------------------------
urls = {
    "homepage": "https://www.bankofamerica.com/",
    "privacy": "https://www.bankofamerica.com/privacy/"
}

# -------------------------------
# 2️⃣ Scrape page text dynamically
# -------------------------------
scraped_pages = {}
for page_name, url in urls.items():
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    text = " ".join(p.get_text() for p in soup.find_all("p"))
    scraped_pages[page_name] = text
    print(f"Extracted text from {page_name}: {len(text)} characters")

# -------------------------------
# 3️⃣ Fetch top US compliance rules dynamically
# -------------------------------
# For demo, using pre-defined top rules (can be replaced with live scraping from NIST, OCC, CFPB)
rules = [
    {"rule": "All customer data must be stored within the United States", "metric": "data_location", "source": "NIST"},
    {"rule": "Access to sensitive data must be restricted to authorized personnel", "metric": "sensitive_access", "source": "OCC"},
    {"rule": "All financial transactions must be logged and auditable", "metric": "transactions_logged", "source": "CFPB"},
    {"rule": "Third-party vendors must sign a data protection agreement", "metric": "third_party_agreement", "source": "FFIEC"}
]
rules_df = pd.DataFrame(rules)
print(f"Extracted {len(rules_df)} rules from {rules_df['source'].nunique()} sources")

# -------------------------------
# 4️⃣ Initialize embedding model
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Device set to use cuda" if embedder.device.type == "cuda" else "Using CPU")

# -------------------------------
# 5️⃣ Extract metrics from text dynamically using simple heuristics/keywords
# -------------------------------
def extract_metrics(text):
    # This is a simplified approach for demo; can be enhanced with LLM/NLP
    metrics = {}
    if "US" in text or "United States" in text:
        metrics["data_location"] = "US"
    else:
        metrics["data_location"] = "EU"
    
    if "authorized" in text:
        metrics["sensitive_access"] = "authorized_only"
    else:
        metrics["sensitive_access"] = "everyone"
    
    if "transaction" in text:
        metrics["transactions_logged"] = "yes"
    else:
        metrics["transactions_logged"] = "no"
    
    if "third-party" in text:
        metrics["third_party_agreement"] = "signed"
    else:
        metrics["third_party_agreement"] = "unsigned"
    
    return metrics

# -------------------------------
# 6️⃣ Evaluate compliance for each metric
# -------------------------------
results = []

def evaluate_compliance(page_name, text, metrics):
    for _, rule in rules_df.iterrows():
        rule_text = rule['rule']
        metric_name = rule['metric']
        rule_source = rule['source']
        metric_value = metrics.get(metric_name, "unknown")
        
        # Compute similarity
        rule_emb = embedder.encode([rule_text], convert_to_tensor=True)
        metric_emb = embedder.encode([str(metric_value)], convert_to_tensor=True)
        score = util.cos_sim(rule_emb, metric_emb).item()
        
        compliant = score >= 0.4  # Threshold for compliance
        
        suggested_action = None
        if not compliant:
            suggested_action = f"Change '{metric_name}' to comply with rule: '{rule_text}'"
        
        results.append({
            "page": page_name,
            "metric": metric_name,
            "value": metric_value,
            "matched_rule": rule_text,
            "rule_source": rule_source,
            "similarity_score": score,
            "compliant": compliant,
            "suggested_action": suggested_action
        })

# -------------------------------
# 7️⃣ Run evaluation on all pages
# -------------------------------
for page_name, page_text in scraped_pages.items():
    metrics = extract_metrics(page_text)
    evaluate_compliance(page_name, page_text, metrics)

# -------------------------------
# 8️⃣ Convert results to DataFrame and display
# -------------------------------
results_df = pd.DataFrame(results)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
results_df
