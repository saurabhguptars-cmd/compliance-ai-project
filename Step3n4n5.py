# =========================
# Step 3 + 4 + 5: Monitoring Agents + Reporting + Colab Display
# =========================

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from IPython.display import display, Markdown

# -------------------------------
# Load Step 1 & 2 outputs
# -------------------------------
clauses_df = pd.read_csv("data/processed/clauses_sample.csv")
clauses = clauses_df['clause'].tolist()
corpus_embeddings = np.load("data/processed/corpus_embeddings.npy")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Simulated applications and metrics
# -------------------------------
applications = {
    "AppA": {"data_location": "EU", "sensitive_access": "authorized_only"},
    "AppB": {"data_location": "US", "sensitive_access": "everyone"},
}

# -------------------------------
# Smart contract rules (text-based)
# -------------------------------
smart_contracts = {
    "data_location": "All customer data must be stored within the European Union",
    "sensitive_access": "Access to sensitive data must be restricted to authorized personnel"
}

# -------------------------------
# Monitoring agent function
# -------------------------------
compliance_reports = {}  # capture reports in memory for Step 5

def monitor_application(app_name, metrics):
    print(f"\nMonitoring {app_name}...")
    alerts = []
    suggested_changes = []
    
    for key, rule_text in smart_contracts.items():
        if key in metrics:
            metric_value = metrics[key]
            
            # Semantic similarity between rule and metric
            rule_emb = embedder.encode([rule_text], convert_to_tensor=True)
            metric_emb = embedder.encode([str(metric_value)], convert_to_tensor=True)
            score = util.cos_sim(rule_emb, metric_emb).item()
            
            # Compliance threshold
            if score < 0.4:
                alert_msg = f"⚠ Non-compliance on {key}: value='{metric_value}' vs rule='{rule_text}' (score={score:.2f})"
                suggested_change = f"Change '{key}' of {app_name} to comply with: '{rule_text}'"
                alerts.append(alert_msg)
                suggested_changes.append(suggested_change)
    
    # Reporting
    if not alerts:
        print("✅ All metrics compliant")
    else:
        print("\n".join(alerts))
        print("\nSuggested Actions:")
        for action in suggested_changes:
            print("-", action)
    
    # Save report for each app
    report = {
        "app_name": app_name,
        "compliant": len(alerts) == 0,
        "alerts": alerts,
        "suggested_changes": suggested_changes
    }
    compliance_reports[app_name] = report  # save in memory
    with open(f"data/processed/{app_name}_compliance_report.json", "w") as f:
        json.dump(report, f, indent=2)

# -------------------------------
# Run agents for all applications
# -------------------------------
for app, metrics in applications.items():
    monitor_application(app, metrics)

print("\nAll compliance reports saved in 'data/processed/'")

# -------------------------------
# Step 5: Display compliance summary in Colab
# -------------------------------
summary_rows = []
for app_name, report in compliance_reports.items():
    status = "✅ Compliant" if report['compliant'] else "⚠ Non-Compliant"
    summary_rows.append({
        "Application": app_name,
        "Status": status,
        "Issues Found": len(report['alerts'])
    })

summary_df = pd.DataFrame(summary_rows)
display(Markdown("## Compliance Summary"))
display(summary_df)

# Detailed issues
for app_name, report in compliance_reports.items():
    if not report['compliant']:
        display(Markdown(f"### {app_name} - Issues"))
        issues_df = pd.DataFrame(report['alerts'], columns=["Alert"])
        suggested_df = pd.DataFrame(report['suggested_changes'], columns=["Suggested Change"])
        display(issues_df)
        display(suggested_df)
