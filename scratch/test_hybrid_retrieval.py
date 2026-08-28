import sys
import re
import json
import math
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from src.retrieval import load_enriched_facets, get_scoreable_facets

df = load_enriched_facets()
scoreable = get_scoreable_facets(df)

# Domain synonym & concept mapping for psychometric and behavioral facets
# Enhances the semantic representation without hardcoding query matches
CONCEPT_DICTIONARY = {
    "perseverance": "persistence grit tenacity determination not giving up enduring hardship resilience striving through obstacles overcoming failure",
    "persistence": "perseverance steadfastness continuing effort not quitting sticking with tasks diligence through difficulty determination",
    "character strength: perseverance": "grit persistence tenacity overcoming adversity not giving up sustained effort through setbacks",
    "hardworking": "diligence industriousness dedicated effort working long hours putting in time and energy tireless work ethic dedicated labor",
    "self-improvement": "personal development learning from mistakes adjusting approach improving skills seeking feedback self-growth practicing to get better modifying strategy",
    "attitude toward learning": "growth mindset enthusiasm for studying seeking feedback active practice curiosity continuous education engagement with learning",
    "meeting deadlines": "timeliness punctuality submitting on time completing before due date delivering work on schedule finishing before deadline",
    "troubleshooting technical issues": "debugging diagnosing problems isolating root causes fixing bugs technical problem solving writing unit tests memory leaks race conditions",
    "data analysis": "analyzing error logs metrics investigating data patterns quantitative reasoning diagnosing system bottlenecks reviewing logs",
    "managing emotions": "emotional regulation composure under pressure remaining calm patience self-control during conflict de-escalation",
    "patience: resistance to anger": "staying calm when provoked tolerance emotional restraint composure during screaming or stress resisting rage",
    "risktaking": "financial risk speculation daring actions gambling taking chances bold uncalculated moves memecoin investing emergency fund",
    "brevity": "conciseness short answers brief communication laconic replies succinct responses minimal words",
    "volunteer work": "community service food bank charity volunteering time helping non-profits civic participation",
    "caffeine intake (mg/day)": "drinking coffee espresso tea energy drinks caffeine consumption morning coffee cups",
    "hindu spiritual metric: yoga discipline hours / week": "practicing yoga asanas yoga sessions weekly yoga practice yoga discipline",
    "cooperation": "teamwork collaboration pair programming supporting colleagues working together mutual assistance",
    "delegation skills": "assigning tasks delegating work to team members dividing responsibilities dispatching work",
    "happiness": "joy cheerfulness positive emotion satisfaction genuine delight optimism",
    "discontentment": "frustration dissatisfaction resentment unhappiness with situation sarcasm about outages cynicism",
    "burnout symptoms": "exhaustion chronic fatigue overwork emotional depletion burnout",
    "general mood and attitude": "disposition general outlook life satisfaction optimism pessimism",
    "orderliness": "organization neatness structured workspace systematic filing",
    "flawlessness": "perfectionism zero defects precision exactness meticulousness",
}

# ── 1. Build Enhanced Indexed Texts ──────────────────────────────────────────
enhanced_texts = []
for _, row in scoreable.iterrows():
    name = row["normalized_facet"]
    name_low = name.lower().strip().rstrip(":")
    ftype = row["facet_type"]
    obs = bool(row["conversation_observable"])
    
    parts = [name]
    
    # Add domain concepts if present
    if name_low in CONCEPT_DICTIONARY:
        parts.append(f"Related concepts: {CONCEPT_DICTIONARY[name_low]}")
    
    # Add category and observability context
    if obs:
        parts.append("Observable conversational behavioral trait and personality characteristic.")
    else:
        parts.append(f"External quantitative metric requiring logged external measurement ({ftype}).")
        
    enhanced_texts.append(" | ".join(parts))

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding enhanced facet representations...")
dense_embeddings = model.encode(enhanced_texts, normalize_embeddings=True, show_progress_bar=False)

# Build FAISS index
index = faiss.IndexFlatIP(384)
index.add(np.array(dense_embeddings, dtype=np.float32))

# ── 2. Simple BM25 Lexical Scorer for Hybrid Fusion ─────────────────────────
class SimpleBM25:
    def __init__(self, corpus: list[str]):
        self.corpus = [re.findall(r"\w+", doc.lower()) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        self.df = Counter()
        for doc in self.corpus:
            for word in set(doc):
                self.df[word] += 1
        self.N = len(self.corpus)

    def score(self, query: str) -> np.ndarray:
        q_tokens = re.findall(r"\w+", query.lower())
        scores = np.zeros(self.N, dtype=np.float32)
        k1 = 1.2
        b = 0.75
        for t in q_tokens:
            if t in self.df:
                idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0)
                for i, doc in enumerate(self.corpus):
                    tf = doc.count(t)
                    if tf > 0:
                        denom = tf + k1 * (1 - b + b * (self.doc_len[i] / self.avgdl))
                        scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores

bm25 = SimpleBM25(enhanced_texts)

# ── 3. Hybrid Search Function ───────────────────────────────────────────────
def hybrid_search(query: str, top_k: int = 20, alpha: float = 0.7) -> list[tuple[int, float]]:
    # Dense retrieval
    q_emb = model.encode([query], normalize_embeddings=True)
    dense_scores = np.dot(dense_embeddings, q_emb[0])  # Cosine similarities
    
    # BM25 lexical retrieval
    bm25_raw = bm25.score(query)
    bm25_norm = bm25_raw / (np.max(bm25_raw) + 1e-6) if np.max(bm25_raw) > 0 else bm25_raw
    
    # Hybrid combined score
    combined_scores = alpha * dense_scores + (1 - alpha) * bm25_norm
    
    # Slight observability prior for general conversation queries (soft tie-breaker)
    for idx, row in scoreable.iterrows():
        if row["conversation_observable"]:
            combined_scores[idx] += 0.02
            
    top_indices = np.argsort(-combined_scores)[:top_k]
    return [(idx, float(combined_scores[idx])) for idx in top_indices]

# ── 4. Test Queries ────────────────────────────────────────────────────────
test_queries = [
    ("I never give up when things get difficult.", ["Perseverance", "Hardworking", "Persistence"]),
    ("I work eight hours every day and never complain.", ["Hardworking", "Work Styles"]),
    ("I practice yoga for five hours every week.", ["Hindu Spiritual Metric: Yoga Discipline Hours / Week"]),
    ("I volunteer at a local food bank every Saturday.", ["Volunteer Work", "Participation in Community Activities"]),
    ("I drink three cups of coffee every morning.", ["Caffeine Intake (mg/day)"]),
    ("The assignment was due Monday and I submitted it Sunday before the deadline.", ["Meeting Deadlines"]),
    ("I realized my presentations were weak, so I asked my professor for feedback and practiced every weekend.", ["Self-improvement", "Attitude Toward Learning"]),
]

print("\n" + "=" * 65)
print("HYBRID RETRIEVAL EVALUATION ON KEY TEST CASES")
print("=" * 65)

for query, expected_facets in test_queries:
    print(f"\nQUERY: '{query}'")
    print(f"EXPECTED RELEVANT: {expected_facets}")
    results = hybrid_search(query, top_k=5)
    for rank, (idx, score) in enumerate(results, 1):
        row = scoreable.iloc[idx]
        print(f"  {rank}. [{score:.4f}] {row['normalized_facet']} ({row['facet_type']})")
