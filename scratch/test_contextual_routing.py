import sys
import re
import math
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from src.retrieval import load_enriched_facets, get_scoreable_facets, build_facet_semantic_text, _tokenize, SimpleBM25

df = load_enriched_facets()
scoreable = get_scoreable_facets(df)

# Domain ontology
DOMAINS = {
    "work_effort": {
        "signals": [r"\b(work\w*|job|shift|overtime|office|project|task\w*|deadlines?|stayed\s+late|long\s+hours|eight\s+hours|twelve\s+hours|labor|diligence|industrious\w*|career|colleague\w*|boss|manager|client|deliver\w*|finish\w*|complet\w*)\b"],
        "facet_patterns": [r"\b(work\w*|hardworking|job|career|labor|industrious|diligence|peer-collaboration|delegat\w*)\b"],
    },
    "persistence_adversity": {
        "signals": [r"\b(fail\w*|rejection|reject\w*|setback\w*|difficult\w*|adversity|obstacle\w*|gave\s+up|give\s+up|giving\s+up|persist\w*|persever\w*|grit|tenacity|tri\w*\s+again|appl\w*\s+again|struggl\w*|surrender\w*|never\s+quit)\b"],
        "facet_patterns": [r"\b(persever\w*|persist\w*|grit|resilience|tenacity|doggedness|striving)\b"],
    },
    "learning_growth": {
        "signals": [r"\b(learn\w*|studi\w*|exam\w*|test\w*|interview\w*|practice\w*|prepar\w*|feedback|improve\w*|weak\w*|presentation\w*|professor|teach\w*|course|grade|coach\w*|skill\w*)\b"],
        "facet_patterns": [r"\b(learning|self-improvement|growth|education|study|skill|achievement|intellect)\b"],
    },
    "deadlines_timeliness": {
        "signals": [r"\b(deadline\w*|due\s+date|due|submit\w*|on\s+time|ahead\s+of\s+schedule|late|punctual\w*|schedule\w*|turn\w*\s+in|deliver\w*\s+by)\b"],
        "facet_patterns": [r"\b(deadline\w*|timeliness|punctuality|submission)\b"],
    },
    "technical_problem_solving": {
        "signals": [r"\b(code|coding|debug\w*|bug\w*|error\s+logs?|race\s+condition|memory\s+leak|bottleneck|server|algorithm|software|tech\w*|system)\b"],
        "facet_patterns": [r"\b(troubleshooting|technical|data\s+analysis|debugging|algorithm)\b"],
    },
    "emotion_patience": {
        "signals": [r"\b(ang\w*|calm|yell\w*|scream\w*|frustrat\w*|patient\w*|composure|temper|rage|screaming|provok\w*|emotion\w*)\b"],
        "facet_patterns": [r"\b(emotion|patience|anger|hostility|calm|composure|aggression)\b"],
    },
    "community_volunteering": {
        "signals": [r"\b(volunteer\w*|food\s+bank|charity|shelter|civic|non-profit|community\s+service|donate|donating)\b"],
        "facet_patterns": [r"\b(volunteer|community|charity|civic)\b"],
    },
    "yoga_physical_practice": {
        "signals": [r"\b(yoga|asanas?|pranayama|vinyasa|stretching\s+routine|yoga\s+class)\b"],
        "facet_patterns": [r"\b(yoga)\b"],
    },
    "spiritual_religious": {
        "signals": [r"\b(sufi|dhikr|zohar|kabbalah|quran|khatam|bhagavad|gita|vrata|kirtan|sukkot|lulav|archon|reiki|meditation|mantra|prayer|synagogue|mosque|temple|scripture|spiritual)\b"],
        "facet_patterns": [r"\b(spiritual|sufi|dhikr|zohar|kabbalah|quran|gita|vrata|kirtan|sukkot|lulav|archon|reiki|meditation|mantra|jewish|islamic|hindu|sikh|buddhist|gnostic)\b"],
    },
    "caffeine_beverage": {
        "signals": [r"\b(coffee|espresso|caffeine|tea|latte|cappuccino|energy\s+drinks?)\b"],
        "facet_patterns": [r"\b(caffeine)\b"],
    },
    "transit_commute": {
        "signals": [r"\b(commute|commuting|subway|bus|metro|public\s+transit|train|transit|driving\s+to\s+work)\b"],
        "facet_patterns": [r"\b(commute|transport)\b"],
    },
}

NICHE_DOMAINS = {
    "yoga_physical_practice",
    "spiritual_religious",
    "caffeine_beverage",
    "transit_commute",
}

def extract_conversational_intents(text: str) -> set[str]:
    text_low = text.lower()
    intents = set()
    for domain, config in DOMAINS.items():
        for pat in config["signals"]:
            if re.search(pat, text_low):
                intents.add(domain)
                break
    return intents

def classify_facet_domains(facet_name: str, defn: str = "") -> set[str]:
    combined = f"{facet_name} {defn}".lower()
    facet_domains = set()
    for domain, config in DOMAINS.items():
        for pat in config["facet_patterns"]:
            if re.search(pat, combined):
                facet_domains.add(domain)
                break
    return facet_domains

facet_domain_tags = [
    classify_facet_domains(row["normalized_facet"], str(row.get("scoring_definition", "")))
    for _, row in scoreable.iterrows()
]

# Build BM25 and Dense embeddings
corpus = [build_facet_semantic_text(row) for _, row in scoreable.iterrows()]
bm25 = SimpleBM25(corpus)

model = SentenceTransformer("all-MiniLM-L6-v2")
dense_embs = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)

def contextual_retrieve(query: str, top_k: int = 10, alpha: float = 0.65) -> list[dict]:
    # 1. Intent extraction
    intents = extract_conversational_intents(query)
    
    # 2. Dense retrieval
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(dense_embs, q_emb)
    
    # 3. BM25 retrieval
    bm25_raw = bm25.score(query)
    bm25_max = np.max(bm25_raw)
    bm25_norm = bm25_raw / (bm25_max + 1e-6) if bm25_max > 0 else bm25_raw
    
    # 4. Contextual combination & reranking
    combined = alpha * dense_scores + (1.0 - alpha) * bm25_norm
    
    for i, row in scoreable.iterrows():
        f_domains = facet_domain_tags[i]
        obs = bool(row.get("conversation_observable", True))
        
        # Observable prior
        if obs:
            combined[i] += 0.03
            
        # Domain intent match boost
        if intents and f_domains and (intents & f_domains):
            combined[i] += 0.15
            
        # Niche domain penalty if NOT in active conversational intents
        niche_unmentioned = f_domains & NICHE_DOMAINS
        if niche_unmentioned and not (intents & niche_unmentioned):
            combined[i] -= 0.20
            
    top_indices = np.argsort(-combined)[:top_k]
    
    results = []
    for idx in top_indices:
        row = scoreable.iloc[idx]
        results.append({
            "normalized_facet": row["normalized_facet"],
            "facet_type": row["facet_type"],
            "score": float(combined[idx]),
            "domains": list(facet_domain_tags[idx]),
        })
    return results, intents

test_cases = [
    ("TEST A", "I work eight hours every day."),
    ("TEST B", "I never give up when things get difficult."),
    ("TEST C", "I always submit my assignments before the deadline."),
    ("TEST D", "I realized my presentations were weak, so I asked my professor for feedback, practiced every weekend, recorded myself speaking, and changed my approach based on the feedback."),
    ("TEST E", "I practice yoga for five hours every week."),
    ("TEST F", "I volunteer at a local food bank every Saturday."),
    ("TEST G", "I failed the interview three times, but I kept preparing after every rejection, practiced my answers, asked for feedback, and applied again."),
    ("USER STAYED LATE", "I had a lot of work this week, so I stayed late every day to make sure everything was finished."),
]

print("=" * 70)
print("CONTEXTUAL INTENT ROUTING RETRIEVAL EVALUATION")
print("=" * 70)

for label, query in test_cases:
    results, intents = contextual_retrieve(query, top_k=6)
    print(f"\n{label}: '{query}'")
    print(f"Detected Intents: {list(intents)}")
    print("Top Retrieved Candidates:")
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. [{r['score']:.4f}] {r['normalized_facet']} ({r['facet_type']}) [Domains: {r['domains']}]")
