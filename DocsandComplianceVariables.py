# ============================================
# POC: Legal & Application Compliance Monitoring
# ============================================

# Install required packages if not already installed
!pip install sentence-transformers transformers pandas beautifulsoup4 requests tqdm openpyxl

# -------------------------
# Imports
# -------------------------
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -------------------------
# Configuration
# -------------------------
# List of document URLs or local files
documents = {
    "LegalDoc1": "https://www.sec.gov/privacy",  # legal doc URL example
    "BankApp1": "https://www.bankofamerica.com",  # application-related page
    # Add more documents here
}

# Compliance queries / rules
compliance_rules = [
    {"rule": "Privacy policy adherence", "threshold": 0.3},
    {"rule": "User consent tracking", "threshold": 0.25},
    {"rule": "Data sharing restrictions", "threshold": 0.3},
    {"rule": "Accessibility compliance", "threshold": 0.2},
]

# Initialize models
print("Loading sentence-transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loading summarization model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# -------------------------
# Helper Functions
# -------------------------
def fetch_text(url):
    """Fetch and clean text from URL or local file"""
    if url.startswith("http"):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        texts = soup.stripped_strings
        return list(texts)
    elif os.path.exists(url):
        with open(url, 'r', encoding='utf-8') as f:
            return f.readlines()
    else:
        return []

def summarize_text(text):
    """Summarize text using transformer model"""
    try:
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return text  # fallback to original text if summarization fails

def calculate_risk(similarity, threshold):
    """Assign risk level based on similarity and threshold"""
    if similarity >= threshold + 0.2:
        return "High"
    elif similarity >= threshold:
        return "Medium"
    else:
        return "Low"

# -------------------------
# Process Documents
# -------------------------
results = []

print("Processing documents and calculating compliance...")

for doc_name, url in tqdm(documents.items()):
    paragraphs = fetch_text(url)
    print(f"Total paragraphs collected from {doc_name}: {len(paragraphs)}")
    paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True)
    
    # Check against all compliance rules
    for rule in compliance_rules:
        rule_embedding = embedding_model.encode(rule['rule'], convert_to_tensor=True)
        for i, para_emb in enumerate(paragraph_embeddings):
            similarity = util.cos_sim(rule_embedding, para_emb).item()
            risk = calculate_risk(similarity, rule['threshold'])
            summary = summarize_text(paragraphs[i])
            missing_actionable = "Yes" if similarity < rule['threshold'] else "No"
            
            results.append({
                "Document": doc_name,
                "Paragraph_ID": i+1,
                "Doc_Type": "Legal" if "sec.gov" in url else "App",
                "Text": paragraphs[i][:200] + ("..." if len(paragraphs[i])>200 else ""),
                "Rule_Checked": rule['rule'],
                "Similarity": round(similarity, 3),
                "Risk_Level": risk,
                "Summary": summary,
                "Missing_Actionable": missing_actionable
            })

# -------------------------
# Create DataFrame & Save
# -------------------------
df = pd.DataFrame(results)
print("\nSample output:")
print(df.head(10))

# Save as CSV and Excel
df.to_csv("compliance_report.csv", index=False)
df.to_excel("compliance_report.xlsx", index=False)

# -------------------------
# HTML Report
# -------------------------
def generate_html_report(df, output_file="compliance_report.html"):
    html = "<html><head><title>Compliance Report</title></head><body>"
    html += "<h1>Compliance Monitoring Report</h1>"
    for doc in df['Document'].unique():
        html += f"<h2>Document: {doc}</h2>"
        sub_df = df[df['Document'] == doc]
        html += "<table border='1' style='border-collapse: collapse;'>"
        html += "<tr><th>Paragraph_ID</th><th>Doc_Type</th><th>Rule_Checked</th><th>Similarity</th><th>Risk_Level</th><th>Summary</th><th>Missing_Actionable</th></tr>"
        for _, row in sub_df.iterrows():
            color = "#FF9999" if row['Risk_Level']=="High" else "#FFF799" if row['Risk_Level']=="Medium" else "#99FF99"
            html += f"<tr style='background-color:{color}'><td>{row['Paragraph_ID']}</td><td>{row['Doc_Type']}</td><td>{row['Rule_Checked']}</td><td>{row['Similarity']}</td><td>{row['Risk_Level']}</td><td>{row['Summary']}</td><td>{row['Missing_Actionable']}</td></tr>"
        html += "</table><br>"
    html += "</body></html>"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved as {output_file}")

generate_html_report(df)

print("\n✅ End-to-end POC completed: CSV, Excel & HTML reports generated.")
