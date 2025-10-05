!pip install requests beautifulsoup4 sentence-transformers geoip2 pandas

import requests, time, socket, ssl, geoip2.database
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ==========================================================
# 1️⃣ Load AI Model for Textual Compliance
# ==========================================================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# 2️⃣ Dynamic Rule Extraction from Official Sources
# ==========================================================
rule_sources = {
    "NIST": "https://www.nist.gov/topics/cybersecurity",
    "OCC": "https://www.occ.gov/news-issuances/bulletins/",
    "CFPB": "https://www.consumerfinance.gov/policy-compliance/rulemaking/"
}

def fetch_rules():
    rules = []
    for name, url in rule_sources.items():
        try:
            res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
                for sent in text.split("."):
                    if any(k in sent.lower() for k in ["data", "security", "privacy", "encryption", "access", "storage"]):
                        rules.append({"rule": sent.strip(), "source": name})
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    print(f"Extracted {len(rules)} rules from {len(rule_sources)} sources")
    return rules[:50]  # keep top 50 rules

rules = fetch_rules()
rule_texts = [r["rule"] for r in rules]
rule_embeddings = model.encode(rule_texts, convert_to_tensor=True)

# ==========================================================
# 3️⃣ Dynamic Bank of America URLs
# ==========================================================
boa_sites = [
    "https://www.bankofamerica.com",
    "https://www.bankofamerica.com/security-center/",
    "https://www.bankofamerica.com/privacy/",
    "https://about.bankofamerica.com/en",
]

def fetch_page_text(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        return " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

scraped_pages = {url: fetch_page_text(url) for url in boa_sites}

# ==========================================================
# 4️⃣ Functional Compliance Checks
# ==========================================================
def check_https(url):
    return url.lower().startswith("https://")

def check_tls_version(url):
    try:
        hostname = url.replace("https://", "").split("/")[0]
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                tls = ssock.version()
                return tls in ["TLSv1.2", "TLSv1.3"], tls
    except Exception as e:
        return False, str(e)

def check_data_location(ip):
    try:
        with geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-Country.mmdb') as reader:
            response = reader.country(ip)
            country = response.country.iso_code
            return country == "US", country
    except Exception:
        return False, "Unknown"

# ==========================================================
# 5️⃣ Evaluation Engine (Textual + Functional)
# ==========================================================
def evaluate_site(url, page_text):
    page_emb = model.encode(page_text[:3000], convert_to_tensor=True)
    sims = util.cos_sim(page_emb, rule_embeddings).mean(dim=0)
    best_idx = sims.argmax().item()
    matched_rule = rules[best_idx]["rule"]
    rule_source = rules[best_idx]["source"]

    https_ok = check_https(url)
    tls_ok, tls_detail = check_tls_version(url)
    try:
        ip = socket.gethostbyname(url.replace("https://", "").replace("http://", "").split("/")[0])
        region_ok, region_detail = check_data_location(ip)
    except Exception:
        ip, region_ok, region_detail = "N/A", False, "Unknown"

    compliant = https_ok and tls_ok and region_ok
    suggestion = "✅ Compliant" if compliant else f"⚠️ Review: {matched_rule[:100]}..."

    return {
        "url": url,
        "ip": ip,
        "tls_version": tls_detail,
        "region": region_detail,
        "https_ok": https_ok,
        "tls_ok": tls_ok,
        "region_ok": region_ok,
        "matched_rule": matched_rule,
        "rule_source": rule_source,
        "overall_compliant": compliant,
        "suggestion": suggestion
    }

results = [evaluate_site(url, text) for url, text in scraped_pages.items()]
df = pd.DataFrame(results)
print(df)

# ==========================================================
# 6️⃣ Real-Time Monitoring Agent (Auto-polling)
# ==========================================================
def monitoring_agent(interval=600):
    print("\n🔁 Starting Compliance Monitoring Agent...\n")
    while True:
        for url, text in scraped_pages.items():
            result = evaluate_site(url, text)
            status = "✅ OK" if result["overall_compliant"] else "🚨 Non-Compliant"
            print(f"[{status}] {url} | {result['suggestion']}")
        print("\nSleeping before next scan...\n")
        time.sleep(interval)  # every 10 mins

# Uncomment to activate real-time agent (manual trigger)
# monitoring_agent(interval=600)
