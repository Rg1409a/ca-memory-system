#!/usr/bin/env python3
"""
Fast domain_failure DAG generation — uses just 5 key papers with minimal context.
Goal: get test data into the CA memory system quickly for benchmarking.
"""

import os, sys, json, re, time
from pypdf import PdfReader
import requests

PDF_DIR = "/home/adc/Knowledge_Base/Papers"
OUTPUT_FILE = "/home/adc/nouse_hermes/test_data/synthetic_test_data.json"
LM_URL = "http://localhost:1234/v1/chat/completions"

# 5 most relevant papers for domain_failure physics
TARGET_PAPERS = [
    "Adhesion_related_failure_mechanisms_in_m.pdf",      # Failure mechanisms overview
    "synthetic_domain_graph.pdf",  # Synthetic test data  
    "Surface_adhesion_and_its_dependence_on_s.pdf",       # Adhesion forces
    "11-004.pdf",                                          # Reducing domain_failure (nanometer films)
    "Influence_of_Thin_Film_Deposition_on_AFM.pdf",       # Surface treatment effects
]

PROMPT = """Extract causal triplets from this domain domain_failure text. Return ONLY JSON:
[{{"subject":"...","relation":"...","object":"..."}}]

{text}"""


def extract_text(path, max_pages=1):
    try:
        reader = PdfReader(path)
        parts = []
        for i in range(min(max_pages, len(reader.pages))):
            t = reader.pages[i].extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts)[:3000] if parts else None
    except:
        return None


def extract_triplets(text):
    try:
        r = requests.post(LM_URL, json={
            "model": "qwen3.6-35b-a3b",
            "messages": [
                {"role":"system","content":"Return ONLY valid JSON array."},
                {"role":"user","content": PROMPT.format(text=text)}
            ],
            "temperature": 0.1, "max_tokens": 1000
        }, timeout=30)
        r.raise_for_status()
        m = re.search(r'\[.*\]', r.json()["choices"][0]["message"]["content"], re.DOTALL)
        return json.loads(m.group()) if m else []
    except Exception as e:
        print(f"  LLM error: {e}")
        sys.stdout.flush()
        return []


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("FAST domain_failure DAG GENERATOR (5 papers)")
    print("=" * 50)
    sys.stdout.flush()
    
    all_triplets = {}
    
    for fname in TARGET_PAPERS:
        path = os.path.join(PDF_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP (missing): {fname}")
            sys.stdout.flush()
            continue
        
        text = extract_text(path)
        if not text:
            print(f"SKIP (no text): {fname}")
            sys.stdout.flush()
            continue
        
        t0 = time.time()
        triplets = extract_triplets(text)
        elapsed = time.time() - t0
        
        all_triplets[fname] = {"count": len(triplets), "time_sec": round(elapsed, 1)}
        print(f"OK [{elapsed:.1f}s]: {fname} -> {len(triplets)} triplets")
        sys.stdout.flush()
        time.sleep(0.5)
    
    # Build DAG
    seen = set()
    unique = []
    nodes, edges = set(), []
    
    for fname, data in all_triplets.items():
        path = os.path.join(PDF_DIR, fname)
        reader = PdfReader(path)
        title = (reader.pages[0].extract_text() or "")[:100]
        
        # Re-extract to get triplets per paper
        text = extract_text(path)
        try:
            r = requests.post(LM_URL, json={
                "model": "qwen3.6-35b-a3b",
                "messages": [
                    {"role":"system","content":"Return ONLY valid JSON array."},
                    {"role":"user","content": PROMPT.format(text=text)}
                ],
                "temperature": 0.1, "max_tokens": 1000
            }, timeout=30)
            r.raise_for_status()
            m = re.search(r'\[.*\]', r.json()["choices"][0]["message"]["content"], re.DOTALL)
            if m:
                for t in json.loads(m.group()):
                    key = (t["subject"].lower().strip(), t["relation"].lower().strip(), t["object"].lower().strip())
                    if key not in seen:
                        seen.add(key)
                        unique.append(t)
                        nodes.add(t["subject"])
                        nodes.add(t["object"])
                        edges.append({"source": t["subject"], "target": t["object"], "label": t["relation"]})
        except:
            pass
    
    output = {
        "metadata": {
            "description": "Causal DAG for domain domain_failure physics - test case for CA memory system",
            "total_unique_triplets": len(unique),
            "unique_nodes": len(nodes),
            "papers_used": TARGET_PAPERS,
            "extraction_stats": all_triplets,
        },
        "dag": {"nodes": sorted(list(nodes)), "edges": edges},
        "triplets": unique,
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"  Triplets: {len(unique)} | Nodes: {len(nodes)} | Edges: {len(edges)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
